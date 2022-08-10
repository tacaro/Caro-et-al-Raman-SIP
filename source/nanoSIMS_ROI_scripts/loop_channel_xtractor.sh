for file in $(ls nanoSIMS_roi/Thy_50_12/*.im) ; do output_file=${file%.*} ; python nanosims-processor.py -i $file -o $output_file -c "2H" -C "1H"; done
