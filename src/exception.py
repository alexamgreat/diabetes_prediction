import sys
import traceback


def error_message_detail(error, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # No active traceback (manual raise)
        return f"Error message: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in file [{file_name}] "
        f"at line [{line_number}] "
        f"message: [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message



   
          
    
    
