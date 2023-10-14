import subprocess

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_cmd(cmd):
    try:
        prog = subprocess.Popen(
            cmd,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            encoding="utf-8",
            shell=True
        )
        prog.wait()
        # std_out, err_out = prog.communicate()
        # if prog.poll() == 0:
        #     logger.info(f"std_out: {std_out}")
        # else:
        #     logger.exception(err_out)
    except Exception as e:
        logger.warning(f"Failed run {cmd} {e}")
        exit(1)
