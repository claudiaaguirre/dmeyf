#Necesita para correr en Google Cloud
#64 GB de memoria RAM
#256 GB de espacio en el disco local
#4 vCPU


#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")

#para poder usarlo en la PC y en la nube sin tener que cambiar la ruta
#cambiar aqui las rutas en su maquina
switch ( Sys.info()[['sysname']],
         Windows = { directory.root  <-  "M:\\" },   #Windows
         Darwin  = { directory.root  <-  "~/dm/" },  #Apple MAC
         Linux   = { directory.root  <-  "~/buckets/b1/" } #Google Cloud
)
#defino la carpeta donde trabajo
setwd( directory.root )
#setwd("~/buckets/b1/crudoB/" )

version  <- "v002"  #cambiar cada vez, asi se tiene versionado del dataset

dataset  <- fread( "./datasets/dataset_epic_v952_exp9_6.csv.gz" )
#dataset  <- copy(  dataset[  , c("numero_de_cliente","foto_mes","clase_ternaria"),  with=FALSE] )
gc()


#leo TODOS los archivos que estan en la carpeta  modelitos
#y hago el join con  dataset  <numero_de_cliente, foto_mes, clase_ternaria>

archivos  <- list.files( pattern="modelitos.csv.gz", path="./modelitos/" )
for( archivo  in archivos )
{
  darchivo  <- fread( paste0("./modelitos/", archivo ) )
  dataset  <- merge( dataset, darchivo, by=c("numero_de_cliente","foto_mes") )
}

gc()

fwrite( dataset,
        file=paste0( "./datasets/dataset_stacking_", version, ".csv.gz"),
        sep="," )

