# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:01:43 2023

@author: abhay.bhandari
"""
import sys
# from src.logger import logging

def error_message_detail(error, exc_info):
    exc_type, exc_value, exc_traceback = exc_info
    file_name = exc_traceback.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_traceback.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)

    def __str__(self):
        return self.args[0]
