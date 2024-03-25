import os

import numpy as np
import streamlit as st

from front_end.model_selection import initialize_state, display_model_selection
from front_end.model_test_data_visual import show_test_data
from util.common import get_proje_root_path
from util.model_meta_info_reading import get_model_list_from_folder_reading


def main():
    model_names = get_model_list_from_folder_reading()
    proje_root = get_proje_root_path()

    initialize_state({
        'selected_model': None,
        # 'expanded': False
    })

    selected_model = display_model_selection(model_names)

    if selected_model:
        preds = np.load(os.path.join(proje_root, "statistic/model_result_list", selected_model, 'pred.npy'))
        trues = np.load(os.path.join(proje_root, "statistic/model_result_list", selected_model, 'true.npy'))
        show_test_data(preds, trues)


if __name__ == '__main__':
    main()
