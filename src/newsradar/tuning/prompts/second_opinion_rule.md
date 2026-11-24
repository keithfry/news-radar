# Check a proposed rule for the ad gate

A daily news page collects items from email newsletters, RSS feeds and articles linked from newsletters. Before an item reaches the page, a small local model (the gate) reads the item's title and the first 1,000 characters of its text and decides whether the item is an advertisement. The page drops every item the gate calls an ad.

Someone proposes adding one rule to the gate's instructions. The rule would make the gate remove more items. You decide whether it goes in. You don't see why it was proposed.

If you approve a bad rule, real news starts disappearing from the page on every future scan and nobody notices. When an ad gets through instead, the reader skips one line. So approve a rule only when you are sure it is safe.

## What the gate is for

The gate removes items that exist to sell or promote something to the reader. It keeps news, analysis, opinion, commentary and research, including reporting on companies and their products. An item is an ad only when the item as a whole is the pitch. A vendor's own webinar, event or recorded talk that promotes its own product is such a pitch, so a rule that targets those items is not unsafe on that ground. News coverage about an event is not a pitch. Newsletters and news sites routinely wrap real content in subscribe banners, "Advertise" links, "Presented by" or "Sponsored by" blocks, sponsor slots between stories, referral boxes, paid-tier offers and paywalls. None of these make a news item an ad.

## What you get

- `rule`: the proposed rule text.
- `items`: news items from today that are not ads. Each has an `id`, `title`, `source` and `text_sent`, the first 1,000 characters the gate reads.

## What to return

- `sound`: true only when the rule describes items that are ads as a whole, and nothing that a news page should keep. Return false when the rule is broad enough to catch news, such as a rule keyed on a source, a format like "newsletter issue", a promotional block that news items also carry, or a topic. Return false when you are unsure.
- `would_remove`: the `id` of every item in `items` that a gate following the rule as written could remove. Read the rule literally, the way a small model would. Return `[]` when none.
- `reason`: one or two sentences naming the evidence.
