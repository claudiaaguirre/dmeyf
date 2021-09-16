#Este LightGBM fue construido  para destronar a quienes desde el inicio utilizaron XGBoost y  LightGBM
#mientras sus compañeros luchaban por correr un rpart

#Con los pibes NO

#limpio la memoria
rm( list=ls() )
gc()

require("data.table")
require("lightgbm")

setwd("/Users/claudia/DMenEyF/")  #establezco la carpeta donde voy a trabajar

#cargo el dataset
dataset_train  <- fread("./datasetsOri/paquete_premium_202009.csv")

#creo la clase_binaria donde en la misma bolsa estan los BAJA+1 y BAJA+2
dataset_train[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]

#Quito el Data Drifting de  "ccajas_transacciones"  "Master_mpagominimo"
#campos_buenos  <- setdiff( colnames(dataset),
#                           c("clase_ternaria", "clase01", "ccajas_transacciones", "Master_mpagominimo" ) )

campos_buenos  <- setdiff(  colnames(dataset_train),  c("clase_ternaria", "clase01",
                                                  "ccajas_transacciones", "Master_mpagominimo",
                                                   "foto_mes", 
                                                    "internet", 
                                                    "mactivos_margen", 
                                                    "mpasivos_margen", 
                                                    "tpaquete1", 
                                                    "mcajeros_propios_descuentos", 
                                                    "mtarjeta_visa_descuentos", 
                                                    "mtarjeta_master_descuentos", 
                                                    "matm_other","tmobile_app",
                                                    "cmobile_app_trx", 
                                                    "Master_Finiciomora") )


#genero el formato requerido por LightGBM

dtrainlgb  <- lgb.Dataset( data=  data.matrix(  dataset_train[ , campos_buenos, with=FALSE]),
                        label= dataset_train[ , clase01])

#Solo uso DOS hiperparametros,  max_bin  y min_data_in_leaf
#Dadme un punto de apoyo y movere el mundo, Arquimedes
modelo  <- lightgbm( data= dtrainlgb,
                     params= list( objective= "binary",
                                   max_bin= 15, #15
                                   min_data_in_leaf= 2000,#4000
                                   learning_rate= 0.01 )  )

#----------------
#Calculo de la ganancia con los datos de entrenamiento:

prediccion_training  <- predict( modelo,  data.matrix(dataset_train[ , campos_buenos, with=FALSE]))

#dataset_train[, c("numero_de_cliente","clase_ternaria")]
ganancia = dataset_train[, sum( (prediccion_training > 0.031) *ifelse( clase01 == 1, 48750, -1250))]
print(ganancia)
#----------------

#cargo el dataset donde aplico el modelo
dapply  <- fread("./datasetsOri/paquete_premium_202011.csv")

#aplico el modelo a los datos nuevos, dapply
prediccion  <- predict( modelo,  data.matrix( dapply[  , campos_buenos, with=FALSE]))

#la probabilidad de corte ya no es 0.025,  sino que 0.031
entrega  <- as.data.table( list( "numero_de_cliente"= dapply[  , numero_de_cliente],
                                 "Predicted"= as.numeric(prediccion > 0.031) ) ) #genero la salida

#genero el archivo para Kaggle
fwrite( entrega, 
        file= "./kaggle/lightgbm_con_los_pibes_no.csv",
        sep=  "," )
