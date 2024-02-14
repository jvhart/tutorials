
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': pendulum.today('UTC').add(days=1),
    'email': ['jarodvhart@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'test_fernet_encryption',
    default_args=default_args,
    description='Validate that Fernet encryption and decryption are functioning as expected',
    schedule=timedelta(days=1),
)

task_1 = BashOperator(
    task_id='create_fernet_token',
    bash_command='cd ~/tutorials/airflow && python3 airflow/scripts/example_02/create_fernet_token.py',
    dag=dag,
)

task_2 = BashOperator(
    task_id='create_files',
    bash_command='cd ~/tutorials/airflow && python3 airflow/scripts/example_02/create_files.py',
    dag=dag,
)

task_3 = BashOperator(
    task_id='encrypt_messages',
    bash_command='cd ~/tutorials/airflow && python3 airflow/scripts/example_02/encrypt_messages.py',
    depends_on_past=True,
    dag=dag,
)

task_4 = BashOperator(
    task_id='validate_decryption',
    bash_command='cd ~/tutorials/airflow && python3 airflow/scripts/example_02/validate_decryption.py',
    depends_on_past=True,
    dag=dag,
)

task_5 = BashOperator(
    task_id='cleanup_files',
    bash_command='cd ~/tutorials/airflow && python3 airflow/scripts/example_02/cleanup_files.py',
    depends_on_past=True,
    dag=dag,
)

task_1 >> task_3
task_2 >> task_3
task_3 >> task_4 >> task_5
