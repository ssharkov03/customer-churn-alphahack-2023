import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import SimpleImputer

class TrendHandler:
    def __init__(self, drop_old_cols=False, apply_handler=True):
        self.m3_trend_cols = ['sum_a_oper', 'cnt_a_oper', 
                              'sum_b_oper', 'cnt_b_oper',
                              'sum_c_oper', 'cnt_c_oper',
                              'sum_deb_d_oper', 'cnt_deb_d_oper',
                              'sum_deb_e_oper', 'cnt_deb_e_oper', 'cnt_days_deb_e_oper',
                              'sum_cred_e_oper', 'cnt_cred_e_oper', 'cnt_days_cred_e_oper',
                              'sum_deb_f_oper', 'cnt_deb_f_oper', 'cnt_days_deb_f_oper',
                              'sum_cred_f_oper', 'cnt_cred_f_oper', 'cnt_days_cred_f_oper',
                              'sum_deb_g_oper', 'cnt_deb_g_oper', 'cnt_days_deb_g_oper',
                              'sum_cred_g_oper', 'cnt_cred_g_oper', 'cnt_days_cred_g_oper',
                              'sum_deb_h_oper', 'cnt_deb_h_oper', 'cnt_days_deb_h_oper',
                              'sum_cred_h_oper', 'cnt_cred_h_oper', 'cnt_days_cred_h_oper']
        self.drop_old_cols = drop_old_cols
        self.y1_trend_cols = ['sum_of_paym']
        self.cols_to_drop = [prefix + "_" + suffix for suffix in ['1m', '3m'] for prefix in self.m3_trend_cols] + \
                            [prefix + "_" + suffix for suffix in ['2m', '6m', '1y'] for prefix in self.y1_trend_cols]
        self.apply_handler = apply_handler

    @staticmethod
    def get_trend_3m(df, column_prefix):
        m3 = df[f'{column_prefix}_3m']
        m1 = df[f'{column_prefix}_1m']
        
        start = (m3 - m1) / 2
        end = m1
        seq_columns = [1, 2, 3]

        seq_data = pd.concat([start, start, end], axis=1)
        seq_data.columns = seq_columns
        seq_data_trend = seq_data[seq_columns].pct_change(axis=1).mean(axis=1)
        return seq_data_trend

    @staticmethod
    def get_trend_1y(df, column_prefix):
        y1 = df[f'{column_prefix}_1y']
        m6 = df[f'{column_prefix}_6m']
        m2 = df[f'{column_prefix}_2m']

        start = (y1 - m6) / 3
        mid = (m6 - m2) / 2
        end = m2
        seq_columns = [1, 2, 3, 4, 5, 6]

        seq_data = pd.concat([start, start, start, mid, mid, end], axis=1)
        seq_data.columns = seq_columns
        seq_data_trend = seq_data[seq_columns].pct_change(axis=1).mean(axis=1)
        return seq_data_trend

    def generate_trend_cols(self, df):
        for m3_prefix in tqdm(self.m3_trend_cols, desc="[TrendHandler] Calculating m3 trends..."):
            df[f'{m3_prefix}_trend'] = self.get_trend_3m(df, column_prefix=m3_prefix)
        for y1_prefix in tqdm(self.y1_trend_cols, desc="[TrendHandler] Calculating m1 trends..."):
            df[f'{y1_prefix}_trend'] = self.get_trend_1y(df, column_prefix=y1_prefix)

        if self.drop_old_cols:
            df.drop(self.cols_to_drop, axis=1, inplace=True)
            print(f"[TrendHandler] Dropped old columns: {self.cols_to_drop}")
    def process(self, df):
        if self.apply_handler:
            self.generate_trend_cols(df)


class OkvedHandler:
    def __init__(self, drop_old_col=True, apply_ohe=False, apply_handler=True):
        self.drop_old_col = drop_old_col
        self.apply_ohe = apply_ohe
        self.apply_handler = apply_handler
    
    @staticmethod
    def okved_to_group(okved):
        if not isinstance(okved, str):
            return "okved_NaN"
        okved_id =int(okved.split(".")[0])
        if 1 <= okved_id <= 3:
            return "okved_A"
        if 5 <= okved_id <= 9:
            return "okved_B"
        if 10 <= okved_id <= 33:
            return "okved_C"
        if okved_id == 35:
            return "okved_D"
        if 36 <= okved_id <= 39:
            return "okved_E"
        if 41 <= okved_id <= 43:
            return "okved_F"
        if 45 <= okved_id <= 47:
            return "okved_G"
        if 49 <= okved_id <= 53:
            return "okved_H"
        if 55 <= okved_id <= 56:
            return "okved_I"
        if 58 <= okved_id <= 63:
            return "okved_J"
        if 64 <= okved_id <= 66:
            return "okved_K"
        if okved_id == 68:
            return "okved_L"
        if 69 <= okved_id <= 75:
            return "okved_M"
        if 77 <= okved_id <= 82:
            return "okved_N"
        if okved_id == 84:
            return "okved_O"
        if okved_id == 85:
            return "okved_P"
        if 86 <= okved_id <= 88:
            return "okved_Q"
        if 90 <= okved_id <= 93:
            return "okved_R"
        if 94 <= okved_id <= 96:
            return "okved_S"
        if 97 <= okved_id <= 98:
            return "okved_T"
        if okved_id == 99:
            return "okved_U"
        return "okved_NaN" 

    def okved_to_groups(self, df):
        okved_groups = df['okved'].apply(self.okved_to_group)
        df['okved_groups'] = okved_groups
        if self.drop_old_col:
            df.drop(['okved'], axis=1, inplace=True)
            print("[OkvedHandler] Dropped old column 'okved'")
        if self.apply_ohe:
            df = pd.concat([pd.get_dummies(okved_groups), df], axis=1).drop(['okved_groups'], axis=1)
            print("[OkvedHandler] One-hot-encoding applied to 'okved_groups'")

    def process(self, df):
        if self.apply_handler:
            print("[OkvedHandler] Applying okved -> okved_groups transform...")
            self.okved_to_groups(df)


class InfoHandler:
    def get_cols_info(df):
        cat_cols = list(df.select_dtypes(exclude=["number","bool_"]).columns) 
        num_cols = list(set(df.columns).difference(set(cat_cols)))
        return {"categorical_cols": cat_cols,
                "numerical_cols": num_cols}

class NanHandler:
    def __init__(self, del_high_miss_cols, impute_strategy, update_nans):
        self.high_missing_values_cols = ['max_end_plan_non_fin_deals', 
                                         'max_start_non_fin_deals', 
                                         'min_start_non_fin_deals', 
                                         'min_end_plan_non_fin_deals', 
                                         'max_start_fin_deals', 
                                         'min_end_fact_fin_deals', 
                                         'min_start_fin_deals', 
                                         'max_end_fact_fin_deals']    
        self.do_del_cols = del_high_miss_cols
        self.do_impute = impute_strategy is not None 
        self.do_update_nans = update_nans
        self.impute_strategy = impute_strategy
        self.convert_to_na_dict = {
            "rko_start_months": -999
        }

        if impute_strategy == 'avg':
            self.imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        
        elif impute_strategy == 'kmeans':
            self.img_num = None
            self.img_cat = None
            raise NotImplementedError
        
        elif impute_strategy == 'iterative':
            self.img_num = None
            self.img_cat = None
            raise NotImplementedError

    def del_na_cols(self, df):
        df.drop(self.high_missing_values_cols, axis=1, inplace=True)

    def impute(self, df, is_train=True):
        cols_info = InfoHandler.get_cols_info(df)
        cat_cols = cols_info['categorical_cols']
        cont_cols = cols_info['numerical_cols']
    
        if is_train:
            df[cont_cols] = self.imp_cat.fit_transform(df[cont_cols])
            df[cat_cols] = self.imp_cat.fit_transform(df[cat_cols])
        else:
            df[cont_cols] = self.imp_cat.transform(df[cont_cols])
            df[cat_cols] = self.imp_cat.transform(df[cat_cols])

    def values_to_na(self, df):
        for column, value in self.convert_to_na_dict.items():
            df[column] = df[column].apply(lambda x: x if x != value else np.nan)
            print(f"[NanHandler] Converted value {value} from column {column} to np.nan")
        # df.replace(None, np.nan, inplace=True)
    
    def process(self, df, is_train=True):
        if df is None:
            print(f"[NanHandler] Provided dataframe is None => Skipping NaN processing")
            return
        if self.do_del_cols:
            self.del_na_cols(df)
            print(f"[NanHandler] Deleting high percentage nan columns: {self.high_missing_values_cols}")
        if self.do_update_nans:
            self.values_to_na(df) 
        if self.do_impute:
            print(f"[NanHandler] Performing NaN imputation with '{self.impute_strategy}' strategy...")
            self.impute(df, is_train=is_train)


class ColumnRemover:
    def __init__(self):
        self.cols_to_remove = ['id']
    def process(self, df):
        df.drop(self.cols_to_remove, axis=1, inplace=True)
        print(f"[ColumnRemover] Dropped columns {self.cols_to_remove}")
