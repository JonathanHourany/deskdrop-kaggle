import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

ARTICLES_DTYPES = {"contentType": "category", "lang": "category"}


def load_data(
    path: Path,
    nrows: int | None = None,
    date_col: str | None = None,
    dtypes: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        nrows=nrows,
        dtype=dtypes,
    )

    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], unit="s")

    return data


if __name__ == "__main__":
    header_col, abstract_col = st.columns((1, 3))

    with header_col:
        """
        # Deskdrop: EDA
        An introductory dive in to Recommender Systems"
        """

    with abstract_col:
        """
        What follows is EDA on the Deskdrop data set; a Kaggle data set useful
        introductions into Recommender Systems with both Collaborative and
        Content-Based Filtering

        """

    """# Shared Articles"""
    articles = Path("data/shared_articles.csv")
    loading_prompt = st.text(f"Loading {articles.name}...")
    articles_df = load_data(
        articles,
        date_col="timestamp",
        dtypes=ARTICLES_DTYPES,
    )
    loading_prompt.text(f"{articles.name} loaded \N{thumbs up sign}")

    """
    ## Raw Data
    This file contains information about the articles shared in the platform.
    Each article has its sharing date (timestamp), the original url, title, content in
    plain text, the article' lang (Portuguese - pt or English - en) and information
    about the user who shared the article (author). Note that the authorID isn't the
    person who *made* the article, just the person who shared it.

    There are two possible event types at a given timestamp:

    CONTENT SHARED: The article was shared in the platform and is available for users.

    CONTENT REMOVED: The article was removed from the platform and not available for
    further recommendation.

    'timestamp' is converted to datetime, while 'lang' and 'contentType'
    are converted to categorial
    """

    articles_df

    """## Data Frame Info"""
    buffer = io.StringIO()
    articles_df.info(verbose=True, buf=buffer, show_counts=True)
    st.text(buffer.getvalue())

    """Data Describe"""
    articles_describe = articles_df.describe(
        include=["category", np.datetime64], datetime_is_numeric=True
    ).to_markdown()
    articles_describe

    (
        articles_cols,
        articles_dtypes,
        articles_val_counts,
        author_val_counts,
    ) = st.columns(
        (
            1,
            1,
            2,
            3,
        )
    )

    with articles_cols:
        "Shared Articles Columns"
        articles_df.columns

    with articles_dtypes:
        "Shared Articles Dtypes"
        st.write(articles_df.dtypes.astype("str"))

    with articles_val_counts:
        "Value Counts on 'lang', 'eventType', and 'contentType'"
        val_counts_df = (
            pd.DataFrame(
                articles_df[["lang", "eventType", "contentType"]]
                .melt(var_name="column", value_name="value")
                .value_counts(dropna=False)
            )
            .rename(columns={0: "counts"})
            .sort_values(by=["column", "counts"], ascending=[True, False])
        )
        val_counts_df

    with author_val_counts:
        "Value Counts on Author types"
        author_counts_df = (
            pd.DataFrame(
                articles_df[["authorRegion", "authorCountry", "authorUserAgent"]]
                .melt(var_name="column", value_name="value")
                .value_counts(dropna=False)
            )
            .rename(columns={0: "counts"})
            .sort_values(by=["column", "counts"], ascending=[True, False])
        )
        author_counts_df

    """## Text Columns"""
    articles_df[["authorCountry", "lang", "url", "title", "text"]]

    """Datetime Frequency"""
    date_select_map = {
        "Month": np.histogram(
            articles_df["timestamp"].dt.month,
            bins=12,
        )[0],
        "Hour": np.histogram(
            articles_df["timestamp"].dt.hour,
            bins=24,
        )[0],
        "Day of Week": np.histogram(
            articles_df["timestamp"].dt.weekday,
            bins=7,
        )[0],
    }
    date_freq_option = st.selectbox(
        "Select a datetime resolution", options=("Month", "Hour", "Day of Week")
    )
    date_hist = date_select_map[date_freq_option]
    st.bar_chart(date_hist)

    """## Duplicates"""
    articles_df[
        [
            "timestamp",
            "eventType",
            "contentId",
            "contentType",
            "authorPersonId",
            "url",
            "title",
        ]
    ][articles_df["title"].duplicated(keep=False)]

    """## Most Common Authors"""
    author_freq = (
        articles_df.groupby(["authorPersonId"]).size().sort_values(ascending=False)
    )
    author_freq = author_freq.to_frame(name="num_articles")
    author_freq["percent"] = (
        author_freq["num_articles"] / author_freq["num_articles"].sum() * 100
    )

    auth_freq_col1, auth_freq_col2 = st.columns((1, 2))

    with auth_freq_col1:
        author_freq

    with auth_freq_col2:
        articles_df[articles_df["authorPersonId"].isin(author_freq.index[:2])]

    """
    ## Notes
    Though author country is missing the majority of the time and probably can't be
    guessed. The second most proflic author, 3609194402293569455 shares in both
    languages.

    Just 3 people make up nearly 30% of the data. Don't expect authorPersonId to be
    an important feature.
    
    ~~a look at the URL and
    language of those rows shows that the majority of the time, the country is likely
    to be the US. Further more, if the language is Portugues, probability is high
    that the author's country is Brazil. A ".br" domain is a dead ringer for this.~~

    People don't use this app on the weekends much.

    Usage picks up around lunch time and slowly drops off after work.

    Feature Engineering: domain name, ~~authorCountry (maybe?)~~. Text fields need to
    be vectorized. Will need to handle CONTENT REMOVED.

    Should take a look at duplicate titles.

    Should take a look at most proflict authors.
    """
