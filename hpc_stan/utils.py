# utils.py
import os
from getpass import getpass
import paramiko

def submit_to_cluster(cluster_address, username=None, password=None):
    username = username if username else input("Enter your username: ")
    password = password if password else getpass("Enter your password: ")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(cluster_address, username=username, password=password)

    # Assuming the Python script is located in the same directory as this Python file
    script_path = os.path.join(os.getcwd(), 'your_python_script.py')
    sftp = ssh.open_sftp()
    sftp.put(script_path, 'remote_python_script.py')
    sftp.close()

    # Modify the command as per your cluster's requirements
    command = "python remote_python_script.py"

    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())

    ssh.close()
