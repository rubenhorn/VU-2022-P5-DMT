
from pathlib import Path
import sys
from numpy import int8, log10, float32, inf
import pandas as pd
import gc

def _append_columns(df, slice):
        df = df.copy()
        for col in slice.columns:
            df[col] = slice[col]
        return df

def _extract_nan(df, columns=None):
    if columns is None:
        print('WARNING: No columns specified. Extracting all columns with NaN values', file=sys.stderr)
        columns = df.columns[df.isna().any()]
    df = df.copy()
    for col in columns:
        # df[col + '_nan_bool'] = df[col].isnull().astype(int8)
        df[col] = df[col].fillna(0)
    return df

def _limit_values(slice, min_value=None, max_value=None):
    df = pd.DataFrame(slice)
    if min_value is not None:
        df[df < min_value] = min_value
    if max_value is not None:
        df[df > max_value] = max_value
    return df

def _normalize_features(input_df, group_key, target_column, take_log10=False):
    # for numerical stability
    epsilon = 1e-4
    if take_log10:
        input_df[target_column] = log10(input_df[target_column] + epsilon)
    methods = ["mean", "std"]

    df = input_df.groupby(group_key).agg({target_column: methods})

    df.columns = df.columns.droplevel()
    col = {}
    for method in methods:
        col[method] = target_column + "_" + method

    df.rename(columns=col, inplace=True)
    df_merge = input_df.merge(df.reset_index(), on=group_key)
    df_merge[target_column + "_norm_by_" + group_key] = (
        df_merge[target_column] - df_merge[target_column + "_mean"]
    ) / df_merge[target_column + "_std"]
    df_merge[target_column + "_norm_by_" + group_key] = df_merge[target_column + "_norm_by_" + group_key]
    df_merge[target_column + "_norm_by_" + group_key].replace([inf, -inf], 0,inplace=True)
    # df_merge = df_merge.drop(labels=[col["mean"], col["std"]], axis=1)

    gc.collect()
    return df_merge[target_column + "_norm_by_" + group_key]

def _get_stat_features(input_df, group_key, target_column):
    methods = ["mean", "std"]
    df = input_df.groupby(group_key).agg({target_column: methods})
    df.columns = df.columns.droplevel()
    col = {}
    for method in methods:
        col[method] = target_column + "_" + method
    df.rename(columns=col, inplace=True)

    gc.collect()
    return df
class Preprocessing:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame()
        out_base_path = (Path(__file__).parent / 'dataset').resolve()
        lookup_book_sum = pd.read_csv(out_base_path / 'prop_booking_freq.csv')
        lookup_position = pd.read_csv(out_base_path / 'prop_position.csv')

        # Property related features
        out = _append_columns(out, X[['prop_location_score1', 'prop_location_score2']])
        out = _append_columns(out, X[['prop_review_score', 'prop_starrating', 'srch_query_affinity_score']])
        out['srch_query_affinity_score'].fillna(-10, inplace=True)
        out = _append_columns(out, X[['prop_brand_bool', 'promotion_flag', 'random_bool']])
        prices = _limit_values(X[['price_usd', 'prop_log_historical_price']], max_value=5000)
        out = _append_columns(out, prices)

        # User related features
        out = _append_columns(out, X[['visitor_hist_adr_usd']])
        out = _append_columns(out, X[['visitor_hist_starrating']])

        # Time related features
        datetime = pd.to_datetime(X['date_time'])
        out['month'] = datetime.dt.month
        out['weekday'] = datetime.dt.weekday
        out = _append_columns(out, X[['srch_booking_window']])

        # Search related features
        out = _append_columns(out, X[[
            'srch_saturday_night_bool', 'orig_destination_distance',
            'srch_adults_count', 'srch_children_count',
            'srch_room_count', 'srch_length_of_stay']])

        # Competitors
        out = _append_columns(out, X[[
            'comp1_rate', 'comp2_rate',
            'comp3_rate', 'comp4_rate',
            'comp5_rate', 'comp6_rate',
            'comp7_rate', 'comp8_rate'
        ]])
        out = _append_columns(out, X[[
            'comp1_inv', 'comp2_inv',
            'comp3_inv', 'comp4_inv',
            'comp5_inv', 'comp6_inv',
            'comp7_inv', 'comp8_inv',
        ]])
        out = _append_columns(out, X[[
            'comp1_rate_percent_diff', 'comp2_rate_percent_diff',
            'comp3_rate_percent_diff', 'comp4_rate_percent_diff',
            'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
            'comp7_rate_percent_diff', 'comp8_rate_percent_diff',
        ]])
        out = _extract_nan(out, columns=[
            'comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate',
            'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate',
            'comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv',
            'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv',
            'comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff',
            'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff'
        ])

        # Normalized features
        out = _append_columns(out, _normalize_features(X, 'srch_id', 'price_usd', True).to_frame().astype(float32))
        out = _append_columns(out, _normalize_features(X, 'srch_id', 'prop_starrating').to_frame().astype(float32))
        out = _append_columns(out, _normalize_features(X, 'prop_id', 'price_usd').to_frame().astype(float32))
        # out = out.drop(labels=['price_usd', 'prop_starrating'], axis = 1)
        # print(f'Number of features: {len(out.columns)}'); exit() # DEBUG only

        # Lookup features
        out = _append_columns(out, X[['prop_id']])
        
        out = out.merge(lookup_book_sum, on='prop_id', how='left')
        out = out.merge(lookup_position, on='prop_id', how='left')
        # out['position_mean'].fillna(25, inplace=True)
        out['no_rand_position_mean'].fillna(25, inplace=True)
        # out['position_std'].fillna(15, inplace=True)
        out['no_rand_position_std'].fillna(15, inplace=True)
        out.drop(['position_mean', 'position_std', 'booking_freq'], axis=1, inplace=True)

        out = _append_columns(out, X[['srch_id']])
        # out = out.merge(_get_stat_features(X, group_key='srch_id', target_column='prop_starrating'), on='srch_id', how='left')
        # out = out.merge(_get_stat_features(X, group_key='srch_id', target_column='prop_review_score'), on='srch_id', how='left')
        # out = out.merge(_get_stat_features(X, group_key='srch_id', target_column='prop_location_score1'), on='srch_id', how='left')
        # out = out.merge(_get_stat_features(X, group_key='srch_id', target_column='prop_location_score2'), on='srch_id', how='left')

        out = out.merge(_get_stat_features(X, group_key='prop_id', target_column='srch_length_of_stay'), on='prop_id', how='left')
        out = out.merge(_get_stat_features(X, group_key='prop_id', target_column='srch_booking_window'), on='prop_id', how='left')
        out = out.merge(_get_stat_features(X, group_key='prop_id', target_column='srch_adults_count'), on='prop_id', how='left')
        out = out.merge(_get_stat_features(X, group_key='prop_id', target_column='srch_children_count'), on='prop_id', how='left')
        out = out.merge(_get_stat_features(X, group_key='prop_id', target_column='srch_room_count'), on='prop_id', how='left')

        out.drop('srch_id', 1, inplace=True)
        out.drop('prop_id', 1, inplace=True)

        cols_nan = [
            'prop_location_score2', 'prop_review_score', 
            'visitor_hist_adr_usd', 'visitor_hist_starrating',
            'orig_destination_distance', 'price_usd_norm_by_srch_id',
            'prop_starrating_norm_by_srch_id','price_usd_norm_by_prop_id',
            'booking_sum',
            # 'prop_starrating_std', 'prop_starrating_mean',
            # 'prop_review_score_std', 'prop_review_score_mean',
            # 'prop_location_score1_std', 'prop_location_score1_mean',
            # 'prop_location_score2_std', 'prop_location_score2_mean',
            'srch_length_of_stay_std', 'srch_length_of_stay_mean',
            'srch_booking_window_std', 'srch_booking_window_mean',
            'srch_adults_count_std', 'srch_adults_count_mean',
            'srch_children_count_std', 'srch_children_count_mean',
            'srch_room_count_std', 'srch_room_count_mean']
        out = _extract_nan(out, columns=cols_nan)

        # Check if there are any columns with NaN values not in cols_nan
        cols_still_with_nan = list(out.columns[out.isna().any()])
        assert len(cols_still_with_nan) == 0, f'Columns with NaN values: { cols_still_with_nan }'

        return out
