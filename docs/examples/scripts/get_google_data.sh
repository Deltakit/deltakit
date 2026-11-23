where=${1:-data}
where_google=$where/examples/google_qmem_experiment/full_dataset

echo "\n1. Creating the folder [$where_google]\n"
mkdir -p $where_google
rm -r $where_google/*

echo "\n2. Downloading the file with data\n"
wget --no-check-certificate "https://zenodo.org/records/6804040/files/google_qec3v5_experiment_data.zip?download=1" -O $where_google/data.zip

echo "\n3. Validating md5 checksum\n"
# mac specific comment:
# brew install rhash coreutils
cat $where_google/data.zip | md5sum -b | grep "a7fd8b481c3087090093106382dc217d"
if [ $? != 0 ]; then
    echo "    Checksum validation failed"
    exit 1
fi

echo "\n3. Unpacking the file\n"
unzip -o $where_google/data.zip -x "repetition_code_*" "surface_code_bZ_d5*" "*/*.dem" "*/layout.svg" "*/*.yml" "*/*_predicted_*" "*/detection_events.b8" "*/*.01" "*_r09_*" "*_r11_*" "*_r13_*" "*_r15_*" "*_r17_*" "*_r19_*" "*_r2*" -d $where_google
# if we want to keep exactly all google data, we may run:
# unzip -o $where_google/data.zip -d $where_google

echo "\n4. Removing original archive\n"
rm $where_google/data.zip

echo "Done!"
