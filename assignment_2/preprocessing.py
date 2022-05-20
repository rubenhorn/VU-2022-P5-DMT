
import sys
from numpy import int8
import pandas as pd


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
        df[col + '_nan_bool'] = df[col].isnull().astype(int8)
        df[col] = df[col].fillna(0)
    return df

def _limit_values(slice, min_value=None, max_value=None):
    df = pd.DataFrame(slice)
    if min_value is not None:
        df[df < min_value] = min_value
    if max_value is not None:
        df[df > max_value] = max_value
    return df

class Preprocessing:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame()

        # Property related features
        out = _append_columns(out, X[['prop_location_score1', 'prop_location_score2']])
        out = _append_columns(out, X[['prop_review_score', 'prop_starrating', 'srch_query_affinity_score']])
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

        # print(f'Number of features: {len(out.columns)}'); exit() # DEBUG only

        cols_nan = [
            'prop_location_score2', 'prop_review_score', 
            'visitor_hist_adr_usd', 'visitor_hist_starrating',
            'orig_destination_distance', 'srch_query_affinity_score']
        out = _extract_nan(out, columns=cols_nan)

        # Check if there are any columns with NaN values not in cols_nan
        cols_still_with_nan = list(out.columns[out.isna().any()])
        assert len(cols_still_with_nan) == 0, f'Columns with NaN values: { cols_still_with_nan }'

        return out
