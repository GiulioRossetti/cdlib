
#This first command creates a template .yaml file from a pipy package that can can be used to generate the coresponding conda package
#It does not work for several packages, including cdlib, karateclub...
#In those cases, I manually created or modified the .yaml file
#We therefore consider that the .yaml files must be updated manually when needed (basically, url and version number)
#------
#conda skeleton pypi *package name*
#------

#build everything for the current platform (very long)
conda config --set anaconda_upload no


#-c allows to specify a necessary channel, if everything is not built locally first it's necessary to add our own channel (currently, yquetzal)
#conda-build bimlpa -c conda-forge
#conda-build demon -c conda-forge
#conda-build eva_lcd -c conda-forge
#conda-build karateclub -c conda-forge
#conda-build nf1 -c conda-forge
#conda-build omega_index_py3 -c conda-forge
#conda-build pquality -c conda-forge

#conda-build count-dict -c conda-forge
#conda-build multivalued-dict -c conda-forge
#conda-build shuffle-graph -c conda-forge
#conda-build aslpaw -c conda-forge

conda-build cdlib -c conda-forge


#convert everything for all platforms. You can check the address as one of the last lines of every conda-build
build_location="~/anaconda3/conda-bld/osx-64/"
conda convert --platform all ~/anaconda3/conda-bld/osx-64/*.tar.bz2 -o outputdir

#upload everything. First time, you'll be asked to provide your login/password on anaconda cloud
#use --force to automatically replace packages
anaconda upload outputdir/*/*tar.bz2