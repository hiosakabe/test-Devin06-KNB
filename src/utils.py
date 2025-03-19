"""
ユーティリティ関数とクラスを提供するモジュール
"""
import time
from contextlib import contextmanager

class Timer:
    """
    with構文で使用するタイマー
    
    使用例:
        with Timer(prefix="処理時間:"):
            実行する処理
    """
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):
        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start_time = None
        self.end_time = None

    @property
    def duration(self):
        if self.end_time is None:
            return 0
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        output_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(output_str)
        else:
            print(output_str)
