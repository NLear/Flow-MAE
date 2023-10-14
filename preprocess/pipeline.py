import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, List, Tuple

from utils.arguments import StageArguments
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class StageBase(ABC):
    total_stage: int = 0

    def __init__(self, args: StageArguments) -> None:
        self.name: str = args.name
        self.output_dir: Path = args.output_dir
        self.stage: int = StageBase.total_stage
        StageBase.total_stage += 1

        self.make_folder(self.output_dir, parents=True)

    @staticmethod
    def make_folder(folder, **kwargs):
        if folder.exists():
            # logger.warning(f"folder {folder} already exits, skipping")
            ...
        else:
            # logger.info(f"making folder {folder}")
            try:
                folder.mkdir(**kwargs)
            except Exception as e:
                raise RuntimeError(f"make folder failure: {e}")

    @staticmethod
    def dump_list(file: Path, lst: List[Path]):
        with open(file, "w") as f:
            for line in lst:
                f.write(str(line) + "\n")

    def run(self):
        logger.info(f"running stage: {self.stage}, {self.name}...")
        self.build_folder()
        self.dump_file_paths()

    @abstractmethod
    def build_folder(self):
        raise NotImplementedError

    @abstractmethod
    def dump_file_paths(self):
        raise NotImplementedError


class CopyStage(StageBase):
    def __init__(self, args: StageArguments) -> None:
        super(CopyStage, self).__init__(args)
        self.src_folder: Path = args.src_folder
        self.dst_folder: Path = args.dst_folder
        self.file2folder: bool = args.file2folder
        self.src_file: Path = args.src_file
        self.dst_file: Path = args.dst_file

        self.src_files: List[Path] = []
        self.dst_files: List[Path] = []

    def build_folder(self):
        self.make_folder(self.dst_folder, parents=True)
        self.__copy_folder_recursive(
            src=self.src_folder, dst=self.dst_folder,
        )

    def __copy_folder_recursive(self, src: Path, dst: Path):
        assert src.is_dir(), f"source dir {src} not exits"
        for path in src.iterdir():
            cur_dst = dst.joinpath(path.relative_to(src))
            if path.is_dir():
                self.make_folder(cur_dst)
                self.__copy_folder_recursive(path, cur_dst)
            elif path.is_file():
                self.src_files.append(path)
                self.dst_files.append(cur_dst)
                if self.file2folder is True:
                    self.make_folder(cur_dst)
            else:
                raise TypeError(f"{path} is not dir or file")

    def dump_file_paths(self):
        self.dump_list(self.src_file, self.src_files)
        self.dump_list(self.dst_file, self.dst_files)


class TraverseStage(StageBase):

    def __init__(self, args):
        super(TraverseStage, self).__init__(args)
        self.src_folder: Path = args.src_folder
        self.src_file: Path = args.src_file

        self.src_files: List[Path] = []

    def build_folder(self):
        self.__traverse_folder(self.src_folder)

    def __traverse_folder(self, folder: Path):
        assert folder.is_dir(), f"root dir {folder} not exits"
        for path in folder.iterdir():
            if path.is_dir():
                self.__traverse_folder(path)
            elif path.is_file():
                self.src_files.append(path)

    def dump_file_paths(self):
        self.dump_list(self.src_file, self.src_files)


class StageFactory:
    stages = [CopyStage, TraverseStage]

    @classmethod
    def get_stage(cls, stage_cls: Type[StageBase], args: StageArguments = None) -> StageBase:
        if isinstance(stage_cls, type):
            assert stage_cls in cls.stages
            stage = stage_cls(args)
        elif isinstance(stage_cls, object):
            stage = stage_cls
            if args is not None:
                logger.warning(f"args is not None when delivering stage object, ignoring args {args}")
        else:
            raise TypeError
        return stage


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.stage: List[Tuple[StageBase, ParallelRun]] = []
        self.__init_pipeline()

    def __init_pipeline(self):
        for stage_cls, arg in self.args:
            stage = StageFactory.get_stage(stage_cls, arg)
            parallel = ParallelRun(arg)
            self.stage.append((stage, parallel))

    def run(self):
        for stage, parallel in self.stage:
            stage.run()
            parallel.run()


class ParallelRun:
    def __init__(self, arg: StageArguments) -> None:
        if arg.category == "CopyStage":
            self.cmd = " ".join(
                [f"parallel --jobs {arg.num_workers} --linebuffer --progress --xapply ",
                 f"{arg.cmd} ",
                 ":::: ", f"{arg.src_file} ",
                 ":::: ", f"{arg.dst_file} "]
            )
        else:
            self.cmd = " ".join(
                [f"parallel --jobs {arg.num_workers} --linebuffer --progress ",
                 f"{arg.cmd} ",
                 ":::: ", f"{arg.src_file} "]
            )

    def run(self):
        parallel_cmd = self.parallel_cmd
        logger.info(parallel_cmd)

        try:
            process = subprocess.Popen(
                parallel_cmd,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                encoding="utf-8",
                shell=True
            )
            process.wait()
        except Exception as e:
            logger.exception(e)

    @property
    def parallel_cmd(self):
        return self.cmd
