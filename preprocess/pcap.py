import shutil
import shutil
import subprocess
from pathlib import Path
from typing import List, Union, Callable

from utils.file_utils import str2path
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def mk_folder(folder: Path, **kwargs):
    if folder.exists():
        logger.warning(f"folder {folder} already exits, skipping")
    else:
        logger.info(f"making folder {folder}")
        folder.mkdir(**kwargs)


def copy_file(src: Path, dst: Path, override=False):
    if dst.exists():
        if override is True:
            logger.warning(f"file {dst} already exits, overriding")
        else:
            logger.warning(f"file {dst} already exits, skip")
            return
    logger.info(f"copy {src} --> {dst}")
    shutil.copy(src, dst)


def copy_folder_recursive(
        src: Path, dst: Path,
        file2folder: bool = False, callback_fn: Callable = None,
        **kwargs
):
    assert src.is_dir(), f"root dir not exits"
    for path in src.iterdir():
        cur_dst = dst.joinpath(path.relative_to(src))
        if path.is_dir():
            mk_folder(cur_dst)
            copy_folder_recursive(path, cur_dst, file2folder, callback_fn, **kwargs)
        elif path.is_file():
            if file2folder is True:
                mk_folder(cur_dst)
            if callback_fn is not None:
                callback_fn(path, cur_dst, **kwargs)


def copy_folder(src: Union[Path, str], dst: Union[Path, str],
                file2folder: bool = False, callback_fn: Callable = None, **kwargs):
    src, dst = str2path(src, dst)
    logger.info(f"src_root_dir {src}")
    logger.info(f"dst_root_dir {dst}")
    mk_folder(dst, parents=True)
    copy_folder_recursive(src=src, dst=dst,
                          file2folder=file2folder, callback_fn=callback_fn,
                          **kwargs)


def traverse_folder_recursive(
        folder: Path,
        callback_fn: Callable = None,
        **kwargs
):
    assert folder.is_dir(), f"root dir {folder} not exits"
    for path in folder.iterdir():
        if path.is_dir():
            traverse_folder_recursive(path, callback_fn, **kwargs)
        elif path.is_file() and callback_fn is not None:
            callback_fn(path, **kwargs)


def _pcap_split_serialize(pcap: Path, dst_dir: Path, splitcap_path: Path):
    cmd = f"mono {str(splitcap_path)} " \
          f"-p 1000 -b 1024000 -d " \
          "-s hostpair " \
          f"-r {str(pcap)} -o {str(dst_dir)}"
    # hostpair  flow
    logger.info(f"running shell command: {cmd}")

    prog = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        shell=True
    )

    std_out, err_out = prog.communicate()
    prog.wait()
    if prog.poll() == 0:
        # logger.info(f"std_out: {std_out}")
        ...
    else:
        logger.exception(err_out)


class FileRecord(object):
    def __init__(self, is_pairs: bool = True):
        self.src: List[Path] = []
        self.dst: List[Path] = []
        self.is_pairs = is_pairs

    def append(self, src: Path, dst: Path = None):
        self.src.append(src)
        if dst is not None:
            assert self.is_pairs is True, f"{dst} invalid"
            self.dst.append(dst)

    def dump(self, src_path: Union[str, Path], dst_path: Union[str, Path] = None):
        # _src, _dst = str2path(src_path, _dst_path)
        with open(src_path, "w") as f:
            for line in self.src:
                f.write(str(line) + "\n")
        if dst_path is not None:
            assert self.is_pairs is True, f"{dst_path} invalid"
            with open(dst_path, "w") as f:
                for line in self.dst:
                    f.write(str(line) + "\n")


def parallel_shell(cmd: str, src_file: str, dst_file: str = None):
    if dst_file is not None:
        cmd = " ".join(
            ["parallel --xapply "
             f"{cmd} "
             ":::: ", f"{src_file} ",
             ":::: ", f"{dst_file} "]
        )
    else:
        cmd = " ".join(
            ["parallel "
             f"{cmd} "
             ":::: ", f"{src_file} "]
        )
    logger.info(cmd)
    try:
        process = subprocess.Popen(
            cmd,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            encoding="utf-8",
            shell=True
        )
        process.wait()
    except Exception as e:
        logger.exception(e)


def extract_tcp_udp(src_file: str, dst_file: str):
    cmd = "bash preprocess/extract.sh "
    parallel_shell(cmd=cmd, src_file=src_file, dst_file=dst_file)


def split_seesionns(src_file: str, dst_file: str, splitcap_path: str):
    cmd = f"mono {splitcap_path} " \
          "-p 1000 -b 1024000 -d " \
          "-s hostpair " \
          "-r {1} -o {2} "
    parallel_shell(cmd=cmd, src_file=src_file, dst_file=dst_file)


def trim_seesionns(src_file: str, dst_file: str):
    cmd = "bash preprocess/trim.sh "
    parallel_shell(cmd=cmd, src_file=src_file, dst_file=dst_file)


def json_seesionns(src_file: str, dst_file: str):
    cmd = "bash preprocess/pcap2json.sh "
    parallel_shell(cmd=cmd, src_file=src_file, dst_file=dst_file)


def rm_small_pcap(src_file: str):
    cmd = "bash preprocess/split_pkt.sh "
    parallel_shell(cmd=cmd, src_file=src_file)


"""
tshark  -r infile -w outfile.pcap -F pcap -Y udp.payload 
tshark  -r infile -w outfile.pcap -F pcap -Y tcp.payload 

editcap  -F pcap -i 3600 -s 176  infile outfile

tshark  -T ek -r infile -c 16 -j tcp.payload > outfile.json
tshark  -T ek -r infile > outfile.json
"""
