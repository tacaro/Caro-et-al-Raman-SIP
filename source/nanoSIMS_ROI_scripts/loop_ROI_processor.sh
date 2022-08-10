for ROI in $(ls nanoSIMS_roi/Mb_00_13/*ROI.png) ; do prefix=$(echo $ROI | sed 's/_f.*$//') ; python nanosims-processor.py --roi $ROI -i ${prefix}.im -o $prefix -c "2H" -C "1H" -n; done
