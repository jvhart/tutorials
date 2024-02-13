
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': pendulum.today('UTC').add(days=-4),
    'email': ['jarodvhart@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=3),
}

def subtraction_filter(x):
    return 4 - int(x)

dag = DAG(
    'my_own_dag_01',
    default_args=default_args,
    description='First example of my own DAG',
    schedule=timedelta(days=1),
    user_defined_filters={'countdown': subtraction_filter},
)

task_1 = BashOperator(
    task_id='calculate_date',
    bash_command='echo "Today is..." && sleep 2 && echo "Calculating..." && sleep 3 && date',
    dag=dag,
)

task_2_templated_command = """
{% for i in range(5) %}
    echo "{{ i }}"
    sleep 1
    echo "{{ params.my_param }}"
{% endfor %}
"""

task_2 = BashOperator(
    task_id='count_up',
    depends_on_past=True,
    bash_command=task_2_templated_command,
    params={'my_param': 'Parameter I passed in'},
    dag=dag,
)

task_2_templated_command = """
{% for i in range(5) %}
    echo "{{ i|countdown }}"
    sleep 1
{% endfor %}
"""

task_3 = BashOperator(
    task_id='count_down',
    depends_on_past=True,
    bash_command=task_2_templated_command,
    dag=dag,
)

task_1 >> task_2 >> task_3
