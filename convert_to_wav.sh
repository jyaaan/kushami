# To run, move to directory with m4a files
# chmod +x convert_to_wav.sh
# ./convert_to_wav.sh
mkdir -p wav_files
for file in *.m4a; do
  output="wav_files/${file%.m4a}.wav"
  echo "Converting $file to $output"
  afconvert -f WAVE -d LEI16 "$file" "$output"
done
