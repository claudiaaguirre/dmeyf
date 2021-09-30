#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")

#cargo los datasets que voy a comparar
setwd("/Users/claudia/DMenEyF/") #establezco la carpeta donde voy a trabajar

datasetA  <- fread( "./datasetsOri/paquete_premium_202009.csv" )
datasetB  <- fread( "./datasetsOri/paquete_premium_202011.csv" )

campos_buenos  <- setdiff(  colnames(datasetA),  c("clase_ternaria", "numero_de_cliente","foto_mes", 
                                                        "internet",
                                                   "tpaquete1",
                                                   "mcuenta_corriente",
                                                   "mcaja_ahorro_dolares",
                                                   "mcuentas_saldo",
                                                   "minversion1_pesos",
                                                   "mpayroll",
                                                   "mpagodeservicios",
                                                   "mcajeros_propios_descuentos",
                                                   "ccajeros_propios_descuentos",
                                                   "ctarjeta_visa_descuentos",
                                                   "mtarjeta_visa_descuentos",
                                                   "ctarjeta_master_descuentos",
                                                   "mtarjeta_master_descuentos",
                                                   "mforex_buy",
                                                   "matm_other",
                                                   "tmobile_app",
                                                   "cmobile_app_trx",
                                                   "Master_Finiciomora",
                                                   "Master_mconsumosdolares",
                                                   "Master_madelantopesos",
                                                   "Master_madelantodolares",
                                                   "Visa_mfinanciacion_limite",
                                                   "Visa_Finiciomora",
                                                   "Visa_msaldodolares",
                                                   "Visa_mpagado",
                                                   "Visa_mpagominimo"
                                                   ) )
dA = datasetA[, mget(campos_buenos)]
dB = datasetB[, mget(campos_buenos)]

#-------------------------------------------------------------
# Model Based
#Tag the data from the batch used to build the current production model as 0.
#Tag the batch of data that we have received since then as 1.
#Develop a model to discriminate between these two labels.
#Evaluate the results and adjust the model if necessary.
#-------------------------------------------------------------

dA$"real_clas"= "A"
dB$"real_clas"= "B"

dAB = rbind(dA, dB)
dAB$real_clas = as.factor(dAB$real_clas)
nrow(dA)
nrow(dB)
nrow(dAB)

#Divido en train y test------------------:
set.seed(621983)
dt = sort(sample(nrow(dAB), nrow(dAB)*.7))
df_train = dAB[dt,]
df_test = dAB[-dt,]
nrow(df_train)
nrow(df_test)
nrow(df_test[df_test$real_clas=="A"])
nrow(df_test[df_test$real_clas=="B"])
#----------------------------------------

#######################
# Usando rpart
#######################
require("rpart")
library(tidyverse)
#genero el modelo
modelo  <- rpart("real_clas ~ .",
                 data= df_train,
                 xval= 0,
                 cp= -1,
                 maxdepth= 10 )

prediccion  <- predict( modelo, df_test , type= "prob") #aplico el modelo

p = as.data.frame(prediccion)
glimpse(p)
p = p %>% mutate(pred_A = A, pred_B = B) %>% select(pred_A,pred_B)
glimpse(p)

#p %>% filter(A>0.9) %>% count()
rbind()

df_test$predict = as.numeric(p)

#prediccion[,"A"] > 0.5, "A", "B"
#df_test[ , Predicted  := as.numeric(prob_baja2 > 0.025) ]

#######################
# Usando lgb
#######################
require("lightgbm")

#https://www.rdocumentation.org/packages/ROCR/versions/1.0-11/topics/prediction
dt_lgb  <- lgb.Dataset( data=  data.matrix(df_train),
                           label= df_train$real_clas)

modelo  <- lightgbm( data= dt_lgb,
                     params= list( objective= "binary",
                              #     max_bin= 15, 
                              #     min_data_in_leaf= 4000,
                              #     learning_rate= 0.05,
                                   num_iterations = 100
                     )  )

prediccion_training  <- predict( modelo,data.matrix(df_test))
print(prediccion_training)

df_test$predict = as.numeric(prediccion_training)

#######################
# Impresión de resultados
#######################

print("-----------COMIENZO DE CORRIDA------------\n")
nrow(df_test[df_test$real_clas == "A" & df_test$predict >= 0.9,]) 
nrow(df_test[df_test$real_clas == "A"]) 
print("--------\n")

nrow(df_test[df_test$real_clas == "B" & df_test$predict >= 0.9,]) 
nrow(df_test[df_test$real_clas == "B",]) 
print("-----------FIN DE CORRIDA------------\n")

df_test[,c("real_clas", "predict")]



#-------------------------------------------------------------
#--------Kolmogorov–Smirnov test (K–S test or KS test)--------
# Do x and y come from the same distribution?
#-------------------------------------------------------------

library(dplyr)

c1 <- c()
c2 <- c()

for( campo in  campos_buenos )
{
  #cat( campo, "  " )
  qa = as.vector(t(dA[, mget(campo)]))
  qb = as.vector(t(dB[, mget(campo)]))
  t = ks.test(qa, qb)
  c1 = append(c1,campo)
  c2 = append(c2,t$p.value)
  #sprintf(fmt = "Test: %f\n", t$p.value) %>% cat()
}

df_result = data.frame(c1, c2)
df_result_order = df_result[order(df_result$c2, decreasing = TRUE),]
df_result_order




