import pandas as pd
import numpy as np
import re

def cleanData(text):
  if not (text is np.nan):
    scripts_remove=["http\S+",        # Xóa địa chỉ web
                    "[^\x00-\x7F]_?", # Ký tự không thể decode về utf-8
                    "@\w+",           # Xóa các mention
                    "#\w+"]           # Xóa các topic

    # Thực hiện các thao tác trong scripts_remove
    for script in scripts_remove:
      text = re.sub(script, "", text)

    #  Xóa các khoảng trắng bị thừa
    text = re.sub("\s+", " ", text)

    # Loại bỏ ký tự trắng bị thừa ở trước và sau
    text = text.strip()
    return text
  return ""