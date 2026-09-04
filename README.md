# skillrecall

Measure how reliably an agent skill gets picked when it is installed next to other skills, and get a short, ranked list of edits that make it get picked more.

A skill is a directory with a `SKILL.md`. The host reads only the `name` and `description` when it decides which skill to use, so those two fields are what an author is really designing. `skillrecall` puts your skill in a room with its competitors, sends in a sample of realistic requests, counts how often yours wins, tries concrete edits to the description, and tells you which ones measurably help.

You edit, run it again, and it tells you what moved.

## Install

```sh
uv tool install skillrecall          # or: pipx install skillrecall
```

To track main instead of a release: `uv tool install git+https://github.com/modiqo/skillrecall`. Release wheels are also attached to each [GitHub release](https://github.com/modiqo/skillrecall/releases).

The core has no dependencies beyond Python 3.11. Optional extras:

| Extra | Adds | Install |
|---|---|---|
| `tokens` | Exact token counts instead of estimates | `uv tool install "skillrecall[tokens]"` |
| `dense` | A local embedding model as a second scorer | `uv tool install "skillrecall[dense]"` |
| `router` | A reference run that asks a model to route a sample of tasks | `uv tool install "skillrecall[router]"` |
| `all` | Everything above | `uv tool install "skillrecall[all]"` |

## Use

```sh
skillrecall assess path/to/my-skill
```

That competes your skill against the closest skills in the public catalog. The skill does not have to be on disk. A skills.sh link, a GitHub link, or an `owner/repo/skill` id works the same way; the skill's files are fetched once and cached:

```sh
skillrecall assess https://skills.sh/owner/repo/my-skill
skillrecall assess https://github.com/owner/repo/tree/main/skills/my-skill
skillrecall assess owner/repo/my-skill
```

A reference that holds several skills is assessed as a collection: a whole repository, a `skills/` directory, or a local folder of skills. Every skill competes against its siblings (plus the catalog unless you say `--no-catalog`) and you get one table, weakest first, plus the pairs inside the collection that take each other's tasks.

```sh
skillrecall assess https://github.com/owner/repo     # every skill in the repo
skillrecall assess ./skills --no-catalog             # local collection, siblings only
```

Add the skills already installed on your machine, or a directory of your own:

```sh
skillrecall assess path/to/my-skill --installed
skillrecall assess path/to/my-skill --corpus ./our-team-skills
skillrecall assess path/to/my-skill --installed --corpus ./our-team-skills --neighbours 60
```

Real requests beat generated ones. If you have them, put one per line in a file:

```sh
skillrecall assess path/to/my-skill --tasks-file requests.txt
```

### What you get

```
Skill: landing-clarity
Competition: 28 other skills (28 from the public catalog)

How often you get picked
  When a task is yours, you are chosen about 6 in 10.
  Confidence: moderate (100 sample tasks).
  In bigger requests that need several skills, you are among the top 3 about 7 in 10.

Who takes your tasks
  landing-page-review (acme/skills)       takes about 1 in 10   44 installs
  landing-page (jezweb/claude-skills)     takes about 1 in 10   38 installs

Whose tasks you take
  landing-pages (someone/skills)          you answer about 1 in 10 of its tasks by mistake

Do these, in order
  1. Say the words people use when they ask for this (headline, subhead); for example add “Also covers headline and subhead.”
     Expected: picked more often, shows up more in bigger tasks.
  2. Remove sentence 4: “Trigger even if the user only pastes a URL, a screenshot, or a block of copy.”
     Expected: shorter by 20 tokens, no loss.

  If you apply them together, the header becomes:
    description: Audit a landing page or hero section for clarity ... Also covers headline and subhead.
  Expected: picked about 9 in 10 (from about 6 in 10); mistaken pickups almost never; 9 tokens shorter.

Structure
  [consider] The description never says what the skill produces.
        Add one clause naming the result, for example “returns a ranked list of fixes”.
```

Every edit shown was applied to a copy of your description and rescored on the same requests. Edits that would hurt are not shown. When you rerun after editing, the report opens with what changed.

### What you get for a collection

```
Collection: https://github.com/owner/repo
12 skills assessed, each against its siblings and the public catalog, in 41s.

Weakest first
  skill               picked  mistaken  top-3  biggest competitor                 first thing to do
  handoff            4 in 10   1 in 10   6/10  claude-handoff (same repo) takes 3 in 10   Add a hand-off to claude-handoff: ...
  grilling           6 in 10   0 in 10   8/10  grill-me (same repo) takes 2 in 10         Remove sentence 3: ...
  ...

Pairs in this repo that take each other's tasks
  claude-handoff   answers about 3 in 10 of handoff's tasks; the reverse is about 2 in 10
```

Add `--detail detailed` to append every skill's full report, or `--format json` for one document with a `skills` array and `sibling_pairs`.

### Output formats

```sh
skillrecall assess my-skill                       # human, simple
skillrecall assess my-skill --detail detailed     # adds sizes, every competitor, every edit tried, sample tasks
skillrecall assess my-skill --explain             # appends how the numbers were measured
skillrecall assess my-skill --format json         # machine-readable, same sections
skillrecall assess my-skill --format json --detail detailed -o report.json
skillrecall history my-skill                      # previous runs for this skill
```

The JSON has a stable top level: `skill`, `competition`, `pickup`, `takes_your_tasks`, `you_take_theirs`, `edits`, `suggested`, `structure`, `since_last_run`. Detailed adds `neighbours`, `all_edits`, `terms`, `sample_tasks`, `scorers`, and `options`.

### Options that matter

| Flag | Default | Meaning |
|---|---|---|
| `--neighbours N` | 40 | How many catalog skills to compete against |
| `--workers N` | 4 | Skills assessed in parallel when the reference is a collection |
| `--tasks N` | 100 | How many of your own requests to sample; more gives a tighter confidence range |
| `--installed` | off | Also compete against every skill installed on this machine |
| `--corpus DIR` | | Compete against a directory of skills; repeatable |
| `--no-catalog` | | Skip the public catalog entirely |
| `--offline` | | Use cached catalog data only, never touch the network |
| `--dense` | off | Add the local embedding scorer (needs the `dense` extra) |
| `--reference` | off | Ask a model to route 30 tasks and report agreement (needs the `router` extra and credentials) |
| `--seed N` | 7 | Sampling seed; same seed and same competitors give the same numbers |
| `--no-save` | | Do not record this run in the history |

Catalog searches and fetched competitor files are cached under `~/.cache/skillrecall` for an hour and a week respectively. Run history lives under `~/.local/state/skillrecall`. Set `GITHUB_TOKEN` to raise the rate limit when many competitors live in repositories with unusual layouts.

## Use from Python

```python
from skillrecall import Options, assess

result = assess(Options(skill_path="path/to/my-skill", include_installed=True))
print(result.recall.value, result.confidence)
for edit in result.edits:
    print(edit.instruction, "->", edit.verdict)
report = result.as_dict(detail=True)
```

## How it works

The method is described step by step in [ALGORITHM.md](ALGORITHM.md). In one paragraph: requests are sampled from your skill's body and from the bodies of its closest competitors; each request is routed across every competitor's name and description by a lexical scorer that treats hand-off sentences as rules, plus an optional embedding scorer; you are "picked" when you score highest; rates carry bootstrap intervals; and every suggested edit is a change that was applied and rescored on the same requests before being shown to you.

## Development

```sh
uv sync --extra dev
uv run pytest
uv build
```

### Releasing

1. Bump the version: `uv version 0.2.0` (edits `pyproject.toml`), commit.
2. Tag and push: `git tag v0.2.0 && git push origin main --tags`.
3. Publish a GitHub release for that tag.

The release workflow checks that the tag matches the package version, runs the tests, builds, publishes to PyPI, and attaches the wheel and sdist to the release. PyPI authentication uses trusted publishing (configure the `pypi` environment and this repository's `publish.yml` as a trusted publisher on the PyPI project), or a `PYPI_API_TOKEN` repository secret if you prefer a token.

## License

MIT
