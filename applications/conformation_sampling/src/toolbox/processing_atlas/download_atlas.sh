for name in $(cat `dirname $0` test_data.csv | grep -v name | awk -F ',' {'print $1'}); do
    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip
    mkdir -p data/atlas/${name}
    unzip ${name}_protein.zip -d data/atlas/${name}
    rm ${name}_protein.zip
done