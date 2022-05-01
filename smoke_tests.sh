echo $(hostname)
whoami
cd /opt/utils
. ./venv/bin/activate
python test_utils.py > tmp_smoke_tests_output.txt 2>&1
echo $? > tmp_smoke_tests_rc.txt
echo "result of smoke tests : $(cat tmp_smoke_tests_rc.txt)"


