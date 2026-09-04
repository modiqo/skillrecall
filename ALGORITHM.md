# How skillrecall measures a skill

This is the step-by-step method behind the report. Nothing here is required reading to use the tool; it exists so the numbers can be audited and reproduced.

## The question being answered

A host holds every installed skill's `name` and `description` in context and, when a request arrives, picks the skill whose entry best matches. The author controls only that entry. So the question is not "how similar is my description to others" but:

> When a request that is mine arrives, and my skill is installed next to its competitors, how often does mine win, and how often do I wrongly win requests that belong to someone else?

Both are rates over requests, so both need a sample of requests and both get a confidence interval.

## Step 1. Load and measure the skill

The skill may be a local directory or a remote reference (a skills.sh link, a GitHub link, or `owner/repo/skill`). A remote skill's text files are listed through the repository tree, downloaded in parallel into the cache, and then treated exactly like a local directory; a catalog copy of the same skill is excluded from its own competition.

`SKILL.md` is split into header and body. The header's `name` and `description` form the resident text, the only thing visible before selection. The body is read after selection; files in the directory are read only when the body points at them.

Measured: resident tokens, description tokens, body lines and tokens, reference files and their sizes, script count, section headings. Tokens are exact when a tokenizer is installed and a deterministic estimate otherwise; the report says which.

### Collections

When the reference holds several skills (a repository, a `skills/` directory, a local folder), each skill is assessed in turn with every sibling added to its competition under the origin `sibling`. The summary sorts skills by pickup rate and lists sibling pairs that win each other's tasks, since a collection's own members are the competitors an author can actually change.

## Step 2. Assemble the competition

Competitors come from three places, tagged by origin:

- `installed`: skill roots on this machine, symlinks resolved and deduplicated.
- `local`: directories passed with `--corpus`.
- `catalog`: the public catalog's search, queried with the full description and with the name plus first sentence. The search ranks by meaning, so it returns competitors that share the task but not the vocabulary. Each hit's `SKILL.md` is fetched from its repository (common paths first, then a tree listing), in parallel, with disk caching.

A competitor with the same name and description as the author's skill is reported as a duplicate install and excluded. Competitors are merged on (name, description).

## Step 3. Sample requests

Requests for the author's skill are drawn from its body, never from its description, so the description is never scored against its own words. Sources in order of reliability: bullets and quoted phrases in usage-style sections ("When to use", "Triggers", "Examples"), quoted phrases anywhere, imperative bullets outside instruction-style sections (procedure, steps, caveats, references are excluded because they address the agent, not the user). Each seed is used once, then reused with request templates ("help me …", "how do I …") to reach the sample size. Sampling is seeded.

If the body yields fewer than five seeds, the description is used and the sample is flagged as weak; the report says so and caps confidence at low. Requests supplied with `--tasks-file` take precedence over generated ones.

Adversarial requests are sampled the same way from the bodies of the closest competitors, which are found by scoring the author's own resident text as if it were a request. Composition requests pair one of the author's requests with one competitor request, to test whether the skill still surfaces when a request needs several skills.

## Step 4. Route every request

Each competitor's resident text is a document. The lexical scorer is BM25 over unigrams and adjacent bigrams with an inverted index, so a request touches only the documents that share a term. BM25 saturates term frequency and normalises document length, which stops a long description padded with example phrases from winning by volume. Inverse document frequency is computed over the assembled competition.

Hand-off sentences are treated as rules, not words. A sentence such as "Not for pricing pages; use pricing-page-audit" names a known competitor and carries a hand-off cue. It is removed from the indexed text, because left in place its words would pull pricing requests toward the author, and it is applied as a yield: when a request matches the sentence's condition terms, the author's score for that request is cut so the named competitor wins. A bare pointer with no condition yields only when the named competitor is already competitive.

With the `dense` extra, a local embedding model scores meaning-level similarity. Per request, the lexical and dense score vectors are each standardised across competitors and averaged, so neither dominates by scale.

The winner of a request is the highest-scoring document. Rank of the author's document is recorded for every own request.

## Step 5. Compute the rates

- Picked: share of own requests the author wins. Reported with mean reciprocal rank.
- Mistaken pickups: share of adversarial requests the author wins.
- In the top k on multi-skill requests: share of composition requests where the author ranks within k (default 3).
- Who takes your tasks: for each competitor, the share of own requests it wins.
- Whose tasks you take: for each competitor, the share of its requests the author wins.

Each rate is a mean of binary outcomes, so its 95% interval is a bootstrap that reduces to binomial resampling: one thousand resamples in under a millisecond, reproducible from the seed. Confidence words map from sample size: very low under 20, low under 60, moderate under 150, high from 150.

## Step 6. Attribute the wins

For every own request, the BM25 contribution of each term in the author's document is accumulated. The top terms are "carrying your wins". Request terms absent from the resident text, appearing in at least two requests and in at most 15% of competitor documents (so generic words never qualify), are "asked for but absent".

## Step 7. Try edits and keep the ones that help

Candidate edits are operations on (name, description):

1. Remove each sentence in turn.
2. Move quoted example phrases out of the description, with the "When to use" section to add to the body.
3. Add a hand-off sentence for each competitor that takes at least 5% of own requests, naming that competitor and its two most distinctive terms.
4. Add a sentence naming the absent terms from step 6.
5. Rename, appending the strongest carrying term, when a competitor's name is identical or nearly so (trigram similarity at or above 0.85).
6. Keep only the first k sentences, for every k.

Each edit is applied to a copy, the author's document is rebuilt (hand-offs reparsed), and steps 4 and 5 rerun on exactly the same requests. The change in each rate is a paired difference with its own bootstrap interval. An edit is accepted when it significantly raises picked, significantly lowers mistaken pickups, or significantly raises top-k, without a significant loss elsewhere; or when it shortens the resident text by at least eight tokens with no significant loss. An edit whose interval for picked lies below zero, or whose mistaken-pickup rise is significant and at least as large as its pickup gain, is marked as harmful and never shown. Accepted edits are ranked by a single score: pickup gain, minus half the mistaken-pickup change, plus a quarter of the top-k change, minus a small cost per added token.

The suggested rewrite stacks accepted edits greedily in rank order, rescoring after each; an edit that stops helping once earlier edits are in place is dropped. The stacked result is reported with its overall change from the baseline.

Sentence removal is matched by sentence text, not position, so stacked edits stay correct.

## Step 8. Structural checks

Independent of routing: description length against the competitors' distribution, a result clause in the description, example-phrase lists in the description, presence of "When to use", procedure, and caveat sections, body length against the 500-line guideline, reference files the body never points at, any single reference above 40,000 tokens, scripts the body never mentions, and header warnings.

## Step 9. Remember and compare

A compact summary (rates, sizes, competitor count, sample size) is stored per skill path. The next run reports movement in plain language: picked up or down, mistaken pickups up or down, description shorter or longer, renamed.

## Optional reference run

With the `router` extra and credentials, a sample of own requests is also routed by a model acting as the host: it sees the request and the competitors' entries in a randomised order and replies with a number. The report gives the model's own pickup rate and its agreement with the local scorer, so the proxy's validity is stated rather than assumed.

## Reproducibility

Same skill, same competitors, same seed, same numbers. Catalog results are cached for an hour and competitor files for a week, so runs within that window use an identical competition. The detailed JSON records the options, scorer labels, catalog status, and sample tasks used.
