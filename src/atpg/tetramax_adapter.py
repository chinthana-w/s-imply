import paramiko
import os

TMAX_HOST = "engr-e132-d02.engr.siu.edu"
TMAX_USER = "siu856558971"  # Updated to match SSH config
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")  # Default SSH key

TMX_HOST_BASE_PATH = "/home/grad/siu856558971/tmax/"
TMAX_CCT_PATH = "dut.v"
TMAX_OUT_PATH = "dut.FAULTS"

STAGING_PATH = "staging"
INIT_SCRIPT_PATH = "src/atpg/init_tmax.tcl"
ATPG_SCRIPT_PATH = "src/atpg/atpg_tmax.tcl"
SAED_LIB_PATH = "data/verilog/saed90nm.v"


class TetramaxAdapter:
    def __init__(self, ssh_key_path=None):
        self.ssh_key_path = ssh_key_path or SSH_KEY_PATH
        self.ssh = self.connect()
        self.initialize()

    def connect(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Load SSH key
            if not os.path.exists(self.ssh_key_path):
                raise FileNotFoundError(f"SSH key not found at {self.ssh_key_path}")

            key = paramiko.Ed25519Key.from_private_key_file(self.ssh_key_path)

            # Connect using SSH key authentication
            ssh.connect(hostname=TMAX_HOST, username=TMAX_USER, pkey=key, timeout=30)
            print(f"Successfully connected to {TMAX_HOST} using SSH key authentication")
            return ssh

        except Exception as e:
            print(f"Failed to connect to {TMAX_HOST}: {str(e)}")
            raise

    def initialize(self):
        # Upload init script to the remote directory
        init_filename = os.path.basename(INIT_SCRIPT_PATH)
        atpg_filename = os.path.basename(ATPG_SCRIPT_PATH)
        self.push_file(INIT_SCRIPT_PATH, TMX_HOST_BASE_PATH + init_filename)
        self.push_file(ATPG_SCRIPT_PATH, TMX_HOST_BASE_PATH + atpg_filename)
        self.ssh.exec_command(f"cd {TMX_HOST_BASE_PATH}")
        self.ssh.exec_command("synopsys")

    def push_file(self, src_path: str, dst_path: str):
        sftp = self.ssh.open_sftp()
        sftp.put(src_path, dst_path)
        sftp.close()

    def load_cct(self, cct_path: str):
        self.push_file(SAED_LIB_PATH, TMX_HOST_BASE_PATH + "saed90nm.v")
        self.push_file(cct_path, TMX_HOST_BASE_PATH + TMAX_CCT_PATH)

    def run_atpg(self, cct_name: str, timeout: int = 60):
        script_lines = get_runner_script_lines(cct_name)
        stdin, stdout, stderr = self.ssh.exec_command("tmax -nogui")

        for line in script_lines:
            stdin.write(line)
            stdin.flush()
            err = stderr.read().decode()
            if err:
                print(f"[ERROR] Command '{line}' failed with error: {err}")
            else:
                out = stdout.read().decode()
                print(f"[INFO] Command '{line}' output: {out}")

        stdin.channel.shutdown_write()

    def pull_file(self, src_path: str, dst_path: str):
        sftp = self.ssh.open_sftp()
        sftp.get(src_path, dst_path)
        sftp.close()

    def close(self):
        """Close the SSH connection"""
        if self.ssh:
            self.ssh.close()
            print("SSH connection closed")

    def __del__(self):
        """Ensure connection is closed when object is destroyed"""
        self.close()

def get_runner_script_lines(cct_name):
    lines = [
        f"set design \"{TMAX_CCT_PATH}\"",
        "read_netlist ./${design}",
        "read_netlist ./saed90nm.v -library",
        f"run_build_model {cct_name}",
        # "set_drc ./dump/${design}.spf",
        "run_drc",
        # "set_faults -model transition",
        "add_faults -all",
        "run_atpg -auto_compression",
        "rm ${design}.FAULTS",
        "write_faults ${design}.FAULTS -all",
        "exit",
    ]
    return lines

if __name__ == "__main__":
    adapter = TetramaxAdapter()
    adapter.load_cct("data/verilog/ISCAS85/c17.v")
    adapter.run_atpg("c17")
    adapter.pull_file(TMX_HOST_BASE_PATH + TMAX_OUT_PATH, "data/pattern/c17.result")
    adapter.close()
