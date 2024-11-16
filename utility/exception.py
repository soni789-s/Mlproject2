import sys
import traceback

def error_message_detail(error_message,error_detail):
    _,_,exc_tb=error_detail.exc_info()
    tb_details=traceback.extract_tb(exc_tb)[-1]
    file_name = tb_details.filename
    line_number = tb_details.lineno
    code_context = tb_details.line.strip()
    return f"Error occurred in file: {file_name}, line: {line_number}, code: {code_context}"


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        self.error_message = error_message_detail(error_message,error_detail)
    def __str__(self):
        return self.error_message