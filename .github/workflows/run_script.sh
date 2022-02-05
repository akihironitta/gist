for i in $(find . -type f -name main.py)
do
  echo "Testing ${i}"
  pip install -r $(dirname ${i})/requirements.txt
  python ${i}
done
