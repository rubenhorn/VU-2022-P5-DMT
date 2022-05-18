
import sys
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
        df[col + '_nan_bool'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    return df

class Preprocessing:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame()

        # Property related features
        out = _append_columns(out, X[['prop_location_score1', 'prop_location_score2']])
        out = _append_columns(out, X[['prop_review_score', 'prop_starrating']])
        out = _append_columns(out, X[['prop_brand_bool', 'promotion_flag', 'random_bool']])
        out = _append_columns(out, X[['price_usd', 'prop_log_historical_price']])

        # User related features
        out = _append_columns(out, X[['gross_bookings_usd', 'visitor_hist_adr_usd']])
        out = _append_columns(out, X[['visitor_hist_starrating']])

        # Booking related features
        # TODO (e.g. date+window, rooms+guests, etc.)

        # Print columns that have NaN values
        # print(list(out.columns[out.isna().any()])); exit()

        cols_nan = ['prop_location_score2', 'prop_review_score', 'gross_bookings_usd', 'visitor_hist_adr_usd', 'visitor_hist_starrating']
        out = _extract_nan(out, columns=cols_nan)

        # Output dimensionality reduction (?)
        # TODO

        return out
