for name in $(cat applications/4d_diffusion/test_data.csv | grep -v name | awk -F ',' {'print $1'}); do
    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip
    mkdir -p dataset/atlas/${name}
    unzip ${name}_protein.zip -d dataset/atlas/${name}
    rm ${name}_protein.zip
done