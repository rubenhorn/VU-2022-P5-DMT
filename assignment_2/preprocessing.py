class Preprocessing:
    # TODO choice of columns only for testing
    X_attrs = ['site_id', 'visitor_location_country_id',
               'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # TODO implement actual preprocessing
        X = X[self.X_attrs]
        X = X.fillna(0)
        return X
