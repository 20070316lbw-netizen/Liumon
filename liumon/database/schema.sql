-- ① 价格表 (prices_cn)
CREATE TABLE IF NOT EXISTS prices_cn (
    ticker      VARCHAR,      -- 股票代码 e.g. "600519.SS"
    date        DATE,         -- 交易日
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,       -- 后复权收盘价
    volume      DOUBLE,
    turn        DOUBLE,       -- 换手率
    ps          DOUBLE,       -- 市销率
    PRIMARY KEY (ticker, date)
);

-- ② 因子截面表 (features_cn)
CREATE TABLE IF NOT EXISTS features_cn (
    ticker              VARCHAR,
    date                DATE,
    index_group         VARCHAR,  -- HS300 / ZZ500
    regime_label        VARCHAR,  -- Bull / Bear
    -- 原始因子
    mom_20d             DOUBLE,
    mom_60d             DOUBLE,
    mom_12m_minus_1m    DOUBLE,
    vol_60d_res         DOUBLE,
    sp_ratio            DOUBLE,
    turn_20d            DOUBLE,
    -- 中性化后排名因子
    mom_20d_rank        DOUBLE,
    mom_60d_rank        DOUBLE,
    mom_12m_minus_1m_rank DOUBLE,
    vol_60d_res_rank    DOUBLE,
    sp_ratio_rank       DOUBLE,
    turn_20d_rank       DOUBLE,
    -- 标签
    label_next_month    DOUBLE,
    label_next_month_rank INTEGER,
    PRIMARY KEY (ticker, date)
);

-- ③ 新闻原始表 (news_raw)
CREATE TABLE IF NOT EXISTS news_raw (
    id          VARCHAR PRIMARY KEY,  -- UUID
    source      VARCHAR,   -- eastmoney / weibo
    pub_time    TIMESTAMP,
    title       TEXT,
    content     TEXT,
    ticker      VARCHAR,   -- 关联股票（可空）
    crawled_at  TIMESTAMP DEFAULT current_timestamp
);

-- ④ 新闻标注表 (news_labeled)
CREATE TABLE IF NOT EXISTS news_labeled (
    id                  VARCHAR PRIMARY KEY,
    news_id             VARCHAR,  -- 关联news_raw.id
    label               VARCHAR,  -- 利好 / 利空 / 中性
    raw_score           DOUBLE,   -- 模型原始得分 [-1, 1]
    calibrated_score    DOUBLE,   -- 校准后得分
    news_type           VARCHAR,  -- individual / macro / festival
    affected_industry   VARCHAR,  -- 受影响行业
    affected_ticker     VARCHAR,  -- 受影响股票（可空）
    reason              TEXT,     -- 标注理由（中文）
    model_version       VARCHAR,  -- 使用的模型版本
    labeled_at          TIMESTAMP DEFAULT current_timestamp
);

-- ⑤ 每日情感汇总 (sentiment_daily)
CREATE TABLE IF NOT EXISTS sentiment_daily (
    ticker      VARCHAR,
    date        DATE,
    news_count  INTEGER,
    avg_score   DOUBLE,   -- 当日平均校准得分
    bull_ratio  DOUBLE,   -- 利好新闻占比
    PRIMARY KEY (ticker, date)
);

-- ⑥ Alpha得分表 (alpha_scores)
CREATE TABLE IF NOT EXISTS alpha_scores (
    ticker      VARCHAR,
    date        DATE,
    score_hs300 DOUBLE,   -- HS300权重下的Alpha得分
    score_zz500 DOUBLE,   -- ZZ500权重下的Alpha得分
    rank_hs300  INTEGER,  -- 截面排名
    rank_zz500  INTEGER,
    PRIMARY KEY (ticker, date)
);

-- ⑦ 实验追踪表 (experiments)
CREATE TABLE IF NOT EXISTS experiments (
    id          VARCHAR PRIMARY KEY,
    timestamp   TIMESTAMP,
    dataset     VARCHAR,
    horizon     VARCHAR,
    features    JSON,
    results     JSON,
    notes       TEXT
);
