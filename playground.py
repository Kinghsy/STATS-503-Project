from sklearn.feature_extraction.text import TfidfVectorizer

train = ["a Chinese a Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "a Tokyo Japan Chinese"]
tv = TfidfVectorizer(
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 3),
    analyzer='word'
)
tv_fit = tv.fit_transform(train)

print(tv_fit.toarray())
print(tv.get_feature_names())
