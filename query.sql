SELECT
  body AS body,
  + subreddit_id AS subreddit_id,
  + subreddit AS subreddit,
FROM
  [fh-bigquery:reddit_comments.2015_05]
ORDER BY
  subreddit ASC
LIMIT
  100
