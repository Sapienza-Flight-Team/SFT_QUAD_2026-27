#!/bin/bash

# Set the folder path to monitor
folder_to_monitor="/home/jetson/sft_cv_2024_2025/data/outs"

# Set the threshold for the number of files
threshold=100

# Set the destination folder on the pen drive
destination_folder="/mnt/pdv_drive/outs"

#clear the output folder
#find "$destination_folder" -empty -exec rm "$folder_to_monitor"/* \;

while true; do
	# Count the number of files in the monitored folder
	file_count=$(ls -l "${folder_to_monitor}"| wc -l)

	# Check if the file count exceeds the threshold
	if [[ ${file_count} -ge ${threshold} ]]; then
		echo "Threshold reached. Downloading files to ${destination_folder}"

		# Use converter Script to move and convert images
		echo "${folder_to_monitor}/img*"
		python3 /home/jetson/sft_cv_2023_2024/tools/image_converter.py ${folder_to_monitor}/img* "${destination_folder}" "${destination_folder}/npy"
		echo "Files downloaded successfully."
	fi

	# Sleep for a while before checking again
	sleep 5
done
