#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")

#cargo los datasets que voy a comparar
setwd("/Users/claudia/DMenEyF/") #establezco la carpeta donde voy a trabajar

datasetA  <- fread( "./datasetsOri/paquete_premium_202009.csv" )
datasetB  <- fread( "./datasetsOri/paquete_premium_202011.csv" )

campos_buenos <-  setdiff(  colnames( datasetA),  
                            c("numero_de_cliente","foto_mes","clase_ternaria") )

#-------------------------------------------------------------
#--------Kolmogorov–Smirnov test (K–S test or KS test)--------
# Do x and y come from the same distribution?
#-------------------------------------------------------------
dA = datasetA[, mget(campos_buenos)]
dB = datasetB[, mget(campos_buenos)]

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

#-------------------------------------------------------------
# Model Based
#Tag the data from the batch used to build the current production model as 0.
#Tag the batch of data that we have received since then as 1.
#Develop a model to discriminate between these two labels.
#Evaluate the results and adjust the model if necessary.
#-------------------------------------------------------------
#######################
# Usando lgb
#######################
dA$"real_clas"= "A"
dB$"real_clas"= "B"

dAB = rbind(dA, dB)
dAB$real_clas = as.factor(dAB$real_clas)
nrow(dA)
nrow(dB)
nrow(dAB)

require("lightgbm")

#https://www.rdocumentation.org/packages/ROCR/versions/1.0-11/topics/prediction
#Since scoring classifiers give relative tendencies towards a negative (low scores) or positive (high scores) class, 
#it has to be declared which class label denotes the negative, and which the positive class. Ideally, 
#labels should be supplied as ordered factor(s), the lower level corresponding to the negative class, 
#the upper level to the positive class. If the labels are factors (unordered), numeric, 
#logical or characters, ordering of the labels is inferred 
#from R's built-in < relation (e.g. 0 < 1, -1 < 1, 'a' < 'b', FALSE < TRUE). 
#Use label.ordering to override this default ordering. Please note that the ordering can be locale-dependent e.g. for 
#character labels '-1' and '1'.

dt_lgb  <- lgb.Dataset( data=  data.matrix(dAB),
                           label= dAB$real_clas)

modelo  <- lightgbm( data= dt_lgb,
                     params= list( objective= "binary",
                                   max_bin= 15, 
                                   min_data_in_leaf= 4000,
                                   learning_rate= 0.05,
                                   num_iterations = 100
                     )  )

prediccion_training  <- predict( modelo,data.matrix(dAB))
print(prediccion_training)



class(prediccion_training)

dAB$predict = prediccion_training
dAB$predict = as.numeric(dAB$predict)
head(dAB[, c("real_clas","predict")])

print("---------------------------------\n")
nrow(dAB[dAB$real_clas == "A" & dAB$predict > 0.9,]) #235354------------
dAB[dAB$real_clas == "A" & dAB$predict > 0.5,]
nrow(dAB[dAB$real_clas == "A"]) #235354       claseA (dA) = 235354

nrow(dAB[dAB$real_clas == "B" & dAB$predict < 0.5,]) #0

nrow(dAB[dAB$real_clas == "B" & dAB$predict >= 0.9,]) #238986------------
nrow(dAB[dAB$real_clas == "B",]) #238986   claseB (dB) = 238986

nrow(dAB[dAB$real_clas == "B" & dAB$predict == 1,]) #0


nrow(dAB[dAB$predict == 1]) #0
nrow(dAB[dAB$predict > 0.99]) #474340 --->Predijo todo como "1"

nrow(dAB[dAB$predict < 0.5]) #0

#confusionMatrix(df_final_train$Category, pred_tr$lda$class)
dAB[dAB$real_clas == 1]

dAB[dAB$predict < 0.99]

#library(caret)
#dAB[, "red_class"= ifelse(dAB$predict < 0.5, 0, 1)]
#dAB[ , red_class  := as.numeric(prob_baja2 > 0.025) ]




