import os
import sys
import logging
from logging import StreamHandler, FileHandler
from logging.handlers import QueueListener, QueueHandler
import multiprocessing as mp

class Log:
    GLOBAL_LISTENNER = None
    GLOBAL_LOG_QUEUE = mp.Queue(-1)
    GLOBAL_LISTENING = False

    @classmethod
    def init_log_queue(cls, level=logging.INFO, path:os.PathLike=None, use_STDOUT:bool=True, use_STDERR:bool=False) -> QueueListener:
        try:
            if cls.GLOBAL_LISTENING:
                if cls.GLOBAL_LISTENNER is not None:
                    cls.GLOBAL_LISTENNER.stop(); del cls.GLOBAL_LISTENNER; print("Global Listener is existing. Stopped.")
                raise Exception("Global Listener already running. Did you forget to stop?")
            handlers = []
            if use_STDOUT:
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(logging.Formatter('[ %(asctime)s | %(name)s | %(levelname)s ] %(message)s'))
                stream_handler.setLevel(Log.get_level(level) if isinstance(level,str) else level); handlers.append(stream_handler);
            if use_STDERR:
                error_handler = logging.StreamHandler(sys.stderr)
                error_handler.setFormatter(logging.Formatter('[ %(asctime)s | %(name)s | %(levelname)s ] %(message)s'))
                error_handler.setLevel(logging.WARNING); handlers.append(error_handler);
            if path is not None:
                file_handeler = logging.FileHandler(f'{path}/global.log')
                file_handeler.setFormatter(logging.Formatter('[ %(asctime)s | %(name)s | %(levelname)s ] %(message)s'))
                file_handeler.setLevel(logging.DEBUG); handlers.append(file_handeler);
            GLOBAL_LISTENNER = logging.handlers.QueueListener(cls.GLOBAL_LOG_QUEUE, *handlers, respect_handler_level=True)
            GLOBAL_LISTENNER.start(); cls.GLOBAL_LISTENING = True; return GLOBAL_LISTENNER;
        except Exception as e: print(f"Error: {e}"); return None;

    @classmethod
    def get_log_queue(cls) -> mp.Queue:
        return cls.GLOBAL_LOG_QUEUE
    
    @classmethod
    def get_logger(cls, name:str, level:str='INFO', path:os.PathLike=None) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(cls.get_level(level))
        handler = QueueHandler(cls.GLOBAL_LOG_QUEUE)
        handler.setLevel(cls.get_level(level))
        handler.setFormatter(logging.Formatter('[ %(asctime)s | %(name)s | %(levelname)s ] %(message)s'))
        logger.addHandler(handler)
        return logger

    @classmethod
    def init_logger(cls, queue, name, level=logging.INFO, path=None):
        print(f'This function is deprecated. Use get_logger instead.')
        queue_handler = logging.handlers.QueueHandler(queue)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('[ %(asctime)s | %(name)s | %(levelname)s ] %(message)s'))
        stream_handler.setLevel(level)
        if path is None: path = os.path.join(os.getcwd(), 'logs')
        file_handeler = logging.FileHandler(f'{path}/{name}.log')
        file_handeler.setFormatter(logging.Formatter('[ %(asctime)s | %(name)s | %(levelname)s ] %(message)s'))
        file_handeler.setLevel(logging.DEBUG)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(queue_handler)
        listener = logging.handlers.QueueListener(queue, stream_handler, file_handeler, respect_handler_level=True)
        listener.start()
        logger.info(f"Logger {name} initialized.")
        return logger, listener
    
    @classmethod
    def get_level(cls, level:str)->int:
        if level == 'DEBUG': return logging.DEBUG;
        elif level == 'INFO': return logging.INFO;
        elif level == 'WARNING': return logging.WARNING;
        elif level == 'ERROR': return logging.ERROR;
        elif level == 'CRITICAL': return logging.CRITICAL;
        else: return logging.INFO;

    @classmethod
    def stop_log_queue(cls, listener : QueueListener) -> bool:
        try: listener.stop(); del listener; cls.GLOBAL_LISTENING = False; return not cls.GLOBAL_LISTENING;
        except Exception as e: print(f"Error: {e}"); return False;