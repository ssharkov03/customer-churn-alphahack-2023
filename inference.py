import os
import argparse
import lightgbm
import numpy as np
import pandas as pd

def main(args):
    
    # Dataset reading and preprocessing 
    test_df = pd.read_parquet(args.path_to_test_parquet)
    cat_cols = [
        'channel_code', 'city', 'city_type',
        'index_city_code', 'ogrn_month', 'ogrn_year',
        'branch_code', 'okved', 'segment'
    ]
    test_df[cat_cols] = test_df[cat_cols].astype("category")
    mean_opp_c_1m, mean_deb_sum_g_1m, mean_deb_sum_g_1m, sum_deb_sum_e_3m, sum_deb_sum_g_3m, cnt_c_oper_3m, cnt_deb_d_oper_1m = 4856.169131547148, 56670.6482158295, 56670.6482158295, 1007570.6071466751, 122025.86948561441, 7.0824815812422495, 0.9976947804997149

    test_df['Ind_sum_c_oper_1m'] = test_df['sum_c_oper_1m'].apply(lambda x: 1 if x < mean_opp_c_1m else 0)
    test_df['Ind_sum_g_oper_1m'] = test_df['sum_deb_g_oper_1m'].apply(lambda x: 1 if x < mean_deb_sum_g_1m else 0)
    test_df['Ind_sum_f_oper_1m'] = test_df['sum_deb_f_oper_1m'].apply(lambda x: 1 if x < mean_deb_sum_g_1m else 0)
    test_df['Ind_sum_e_oper_3m'] = test_df['sum_deb_e_oper_3m'].apply(lambda x: 1 if x < sum_deb_sum_e_3m else 0)
    test_df['Ind_sum_g_oper_3m'] = test_df['sum_deb_g_oper_3m'].apply(lambda x: 1 if x < sum_deb_sum_g_3m else 0)
    test_df['Ind_cnt_c_oper_3m'] = test_df['cnt_c_oper_3m'].apply(lambda x: 1 if x < cnt_c_oper_3m else 0)
    test_df['Ind_cnt_d_oper_1m'] = test_df['cnt_deb_d_oper_3m'].apply(lambda x: 1 if x < cnt_deb_d_oper_1m  else 0)
    test_df['max_end_fact_fin_deals_isna'] = test_df['max_end_fact_fin_deals'].isna()
    test_df['max_end_plan_non_fin_deals_isna'] = test_df['max_end_plan_non_fin_deals'].isna()
    test_df['max_start_fin_deals_isna'] = test_df['max_start_fin_deals'].isna()
    test_df['max_start_non_fin_deals_isna'] = test_df['max_start_non_fin_deals'].isna()
    test_df['min_end_fact_fin_deals_isna'] = test_df['min_end_fact_fin_deals'].isna()
    test_df['min_end_plan_non_fin_deals_isna'] = test_df['min_end_plan_non_fin_deals'].isna()
    test_df['min_start_fin_deals_isna'] = test_df['min_start_fin_deals'].isna()
    test_df['min_start_non_fin_deals_isna'] = test_df['min_start_non_fin_deals'].isna()

    submission = test_df[['id']]
    X = test_df.drop(['id'], axis=1)

    # Load models and infer
    models = [lightgbm.Booster(model_file=os.path.join(args.path_to_models_dir, x)) for x in os.listdir(args.path_to_models_dir)]

    probas = np.zeros(len(X))
    for model in models:
        probas += model.predict(X) / len(models)

    # Save submission
    submission['score'] = probas
    submission.to_csv("ES_submission.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_models_dir", default="models", help="full or absolute path to 'models' directory")
    parser.add_argument("--path_to_test_parquet", required=True, help="full or absolute path to parquet file with test data")
    parser.add_argument("--verbose", action='store_true', help="whether to produce logs")

    args = parser.parse_args()
    main(args)