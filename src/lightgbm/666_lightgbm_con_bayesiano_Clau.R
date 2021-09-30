#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")
require("yaml")

require("parallel")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO") #PAQUETE DE ESTIMACIÓN BAYESIANA

require("data.table")
require("lightgbm")

#defino la carpeta donde trabajo
setwd("/Users/claudia/DMenEyF/") 

kexperimento  <- NA   #NA si se corre la primera vez, un valor concreto si es para continuar procesando

kscript           <- "666_lightGBM"
karch_generacion  <- "./datasetsOri/paquete_premium_202009.csv"
karch_aplicacion  <- "./datasetsOri/paquete_premium_202011.csv"
kBO_iter    <-  150   #cantidad de iteraciones de la Optimizacion Bayesiana


#max_bin: max number of bins that feature values will be bucketed in
#small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
#LightGBM will auto compress memory according to max_bin. For example, LightGBM will use uint8_t for feature value if max_bin=255

#min_data_in_leaf: minimal number of data in one leaf. Can be used to deal with over-fitting.

hs  <- makeParamSet(
  makeNumericParam("learning_rate"       , lower= 0.05   , upper=    0.1),
  makeIntegerParam("max_bin" , lower=  8L  , upper= 255L),  
  makeIntegerParam("min_data_in_leaf", lower=  100L  , upper= 4000L)) 

ksemilla_azar  <- 999979

#------------------------------------------------------------------------------
#Funcion que lleva el registro de los experimentos
get_experimento  <- function()
{
  if( !file.exists( "./maestro.yaml" ) )  cat( file="./maestro.yaml", "experimento: 2000" )
  
  exp  <- read_yaml( "./maestro.yaml" )
  experimento_actual  <- exp$experimento
  
  exp$experimento  <- as.integer(exp$experimento + 1)
  Sys.chmod( "./maestro.yaml", mode = "0644", use_umask = TRUE)
  write_yaml( exp, "./maestro.yaml" )
  Sys.chmod( "./maestro.yaml", mode = "0444", use_umask = TRUE) #dejo el archivo readonly
  
  return( experimento_actual )
}
#------------------------------------------------------------------------------
#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos
loguear  <- function( reg, arch=NA, folder="./work/", ext=".txt", verbose=TRUE )
{
  archivo  <- arch
  if( is.na(arch) )  archivo  <- paste0(  folder, substitute( reg), ext )
  
  if( !file.exists( archivo ) )  #Escribo los titulos
  {
    linea  <- paste0( "fecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )
    
    cat( linea, file=archivo )
  }
  
  linea  <- paste0( format(Sys.time(), "%Y%m%d %H%M%S"),  "\t",     #la fecha y hora
                    gsub( ", ", "\t", toString( reg ) ),  "\n" )
  
  cat( linea, file=archivo, append=TRUE )  #grabo al archivo
  
  if( verbose )  cat( linea )   #imprimo por pantalla
}
#------------------------------------------------------------------------------
#funcion para particionar, es la que Andres reemplaza con caret
particionar  <- function( data, division, agrupa="", campo="fold", start=1, seed=NA )
{
  if( !is.na( seed)  )   set.seed( seed )
  
  bloque  <- unlist( mapply(  function(x,y) { rep( y, x ) }, division, seq( from=start, length.out=length(division) )  ) )
  
  data[ , (campo) :=  sample( rep( bloque, ceiling(.N/length(bloque))) )[1:.N],
        by= agrupa ]
}
#------------------------------------------------------------------------------
lightgbm_simple  <- function( fold_test, pdata, param)
{
  set.seed(ksemilla_azar)
  
  campos = append(campos_buenos,"fold")
  
  #Genero el formato requerido por LightGBM --------
  dtrainlgb  <- lgb.Dataset( data=  data.matrix(pdata[fold!= fold_test , campos_buenos, with=FALSE]),
                             label= pdata[fold!= fold_test, clase01])
  #-------------------------------------------------
  modelo  <- lightgbm( data= dtrainlgb,
                       params= list( objective= "binary",
                                     max_bin= param$max_bin,
                                     min_data_in_leaf= param$min_data_in_leaf,
                                     learning_rate= param$learning_rate))
  
  #modelo  <- lightgbm( data= dtrainlgb,
  #                     params= list( objective= "binary",
  #                                   max_bin= 255,
  #                                   min_data_in_leaf= 4000,
  #                                   learning_rate= 0.05))
  
  prediccion  <- predict( modelo, data.matrix(pdata[ fold==fold_test, campos_buenos, with=FALSE]))

  #----------------------
  ganancia_testing = pdata[fold==fold_test, sum( (prediccion > 0.031) *ifelse( clase01 == 1, 48750, -1250))]
  #----------------------

  return( ganancia_testing )
}
#-------------------------------------------------------------------------------------
crossValidation  <- function( data,param, qfolds, pagrupa, semilla )
{
  divi  <- rep( 1, qfolds )
  particionar( data, divi, seed=semilla, agrupa=pagrupa )
  
  ganancias  <- mcmapply( lightgbm_simple, 
                          seq(qfolds), # 1 2 3 4 5  
                          MoreArgs= list(data,param), 
                          SIMPLIFY= FALSE,
                          mc.cores= 1 )   #se puede subir a 5 si posee Linux o Mac OS
  
  data[ , fold := NULL ]
  #devuelvo la primer ganancia y el promedio
  return( mean( unlist( ganancias )) *  qfolds )   #aqui normalizo la ganancia
}
#------------------------------------------------------------------------------
#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales
EstimarGananciaLightGbm  <- function( x )
{
  GLOBAL_iteracion  <<-  GLOBAL_iteracion + 1
  xval_folds  <- 5

  ganancia  <-  crossValidation( dataset_train, 
                                         param=x, 
                                        qfolds= xval_folds, 
                                        pagrupa= "clase01", 
                                        semilla=ksemilla_azar)
  
  #si tengo una ganancia superadora, genero el archivo para Kaggle
  if(  ganancia > GLOBAL_ganancia_max )
  {
    GLOBAL_ganancia_max <<-  ganancia  #asigno la nueva maxima ganancia
    
    #Genero el formato requerido por LightGBM --------
    dtrainlgb1  <- lgb.Dataset( data=  data.matrix( dataset_train[ , campos_buenos, with=FALSE]),
                               label= dataset_train[ , clase01])
  
    modelo  <- lightgbm( data= dtrainlgb1,
                         params= list( objective= "binary",
                                       max_bin= x$max_bin, #15
                                       min_data_in_leaf= x$min_data_in_leaf,#4000
                                       learning_rate= x$learning_rate))
    
    #genero el vector con la prediccion, la probabilidad de ser positivo
    prediccion  <- predict( modelo, data.matrix(dapply[  , campos_buenos, with=FALSE]))
    
    #la probabilidad de corte ya no es 0.025,  sino que 0.031
    entrega  <- as.data.table( list( "numero_de_cliente"= dapply[  , numero_de_cliente],
                                     "Predicted"= as.numeric(prediccion > 0.031) ) ) #genero la salida
    
    #genero el archivo para Kaggle
    fwrite( entrega, 
            file= paste0(kkaggle, GLOBAL_iteracion, ".csv" ), sep=  ",") #file= "./kaggle/lightgbm_con_los_pibes_no.csv",
            
  }
  #logueo 
  xx  <- x
  xx$xval_folds  <-  xval_folds
  xx$ganancia  <- ganancia
  loguear( xx,  arch= klog )
  
  return( ganancia )
}
################# Comienzo del programa ######################

if( is.na(kexperimento ) )   kexperimento <- get_experimento()  #creo el experimento

#en estos archivos quedan los resultados
kbayesiana  <- paste0("./work/E",  kexperimento, "_lightgbm.RDATA" )
klog        <- paste0("./work/E",  kexperimento, "_lightgbm_log.txt" )
kkaggle     <- paste0("./kaggle/E",kexperimento, "_lightgbm_kaggle_" )


GLOBAL_ganancia_max  <-  -Inf
GLOBAL_iteracion  <- 0

if( file.exists(klog) )
{
  tabla_log  <- fread( klog)
  GLOBAL_iteracion  <- nrow( tabla_log ) -1
  GLOBAL_ganancia_max  <-  tabla_log[ , max(ganancia) ]
}
#-----------------------------------------------------------
#cargo el dataset
dataset_train  <- fread("./datasetsOri/paquete_premium_202009.csv")

#creo la clase_binaria donde en la misma bolsa estan los BAJA+1 y BAJA+2
dataset_train[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]

#Quito el Data Drifting de  "ccajas_transacciones"  "Master_mpagominimo"
campos_buenos  <- setdiff(  colnames(dataset_train),  c("clase_ternaria", "clase01",
                                                  "foto_mes", 
                                                  "internet", 
                                                  "mactivos_margen", 
                                                  "mpasivos_margen", 
                                                  "tpaquete1", 
                                                  "mcajeros_propios_descuentos", 
                                                  "mtarjeta_visa_descuentos", 
                                                  "mtarjeta_master_descuentos", 
                                                  "matm_other",
                                                  "tmobile_app",
                                                  "cmobile_app_trx", 
                                                  "Master_Finiciomora") )

#cargo el dataset donde aplico el modelo
dapply  <- fread("./datasetsOri/paquete_premium_202011.csv")

#-----------------------------------------------------------
#Aqui comienza la configuracion de la Bayesian Optimization
#-----------------------------------------------------------
configureMlr( show.learner.output = FALSE)

funcion_optimizar  <- EstimarGananciaLightGbm

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo
obj.fun  <- makeSingleObjectiveFunction(
  fn=       funcion_optimizar,
  minimize= FALSE,   #estoy Maximizando la ganancia
  noisy=    TRUE,
  par.set=  hs,
  has.simple.signature = FALSE
)

ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI())

surr.km  <-  makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))

#inicio la optimizacion bayesiana
if(!file.exists(kbayesiana)) {
  print("entra")
  run  <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else  run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista

quit( save="no" )





