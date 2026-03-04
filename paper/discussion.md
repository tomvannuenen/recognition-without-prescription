# Discussion

## A Framework for Epistemic-Deontic Decoupling in AI-Mediated Advice

Our findings reveal a systematic pattern across all four LLMs: models demonstrate high epistemic engagement — identifying problems, naming dynamics, deploying diagnostic vocabulary — while simultaneously exhibiting low deontic engagement — avoiding directives, hedging on action, and withholding the prescriptions that community members endorse. We propose that this pattern, which we have called *recognition without prescription*, reflects a broader structural feature of LLM-mediated social interaction that can be understood through the relationship between epistemic and deontic engagement.

Drawing on Stevanovic and Peräkylä's (2012) distinction between epistemic and deontic authority in interaction, we propose a two-dimensional framework (Figure X) that maps advice-giving practices along these axes. Epistemic engagement refers to the degree to which an advice-giver demonstrates understanding of the situation — recognizing harm, identifying patterns, deploying relevant concepts. Deontic engagement refers to the degree to which an advice-giver exercises the authority to direct action — telling someone what to do, granting permission, or endorsing a course of action.

```
                         Deontic Engagement
                      Low                High
                 ┌─────────────┬──────────────────┐
            High │ RECOGNITION │  COMMUNITY       │
 Epistemic       │ WITHOUT     │  ADVICE          │
 Engagement      │ PRESCRIPTION│                  │
                 │             │                  │
                 ├─────────────┼──────────────────┤
            Low  │ GENERIC     │  UNSOLICITED     │
                 │ RESPONSE    │  DIRECTIVE        │
                 │             │                  │
                 └─────────────┴──────────────────┘

 Top-left:     High epistemic, low deontic (LLMs)
 Top-right:    High epistemic, high deontic (r/relationship_advice community)
 Bottom-left:  Low epistemic, low deontic (platitudes, generic comfort)
 Bottom-right: Low epistemic, high deontic (moralizing, unsolicited judgment)
```

**Figure X.** Epistemic-deontic engagement framework for advice-giving practices. LLM-generated advice clusters in the top-left quadrant; community-endorsed human advice clusters in the top-right.

This framework makes visible two key claims.

### Epistemic and deontic engagement are independent dimensions that can decouple

In ordinary community advice-giving, epistemic and deontic engagement tend to co-occur: recognizing that a partner's behaviour is abusive typically leads to recommending that the advice-seeker leave. Our cross-model agreement analysis confirms this coupling in human advice, where leave ratio tracks closely with diagnostic vocabulary. In LLM output, however, these dimensions come apart. Models deploy therapeutic and diagnostic language at rates comparable to or exceeding human advice — they *see* the problem — but their deontic engagement, measured through leave orientation, permission-granting speech acts, and imperative forms, is systematically suppressed. The obvious-cases analysis makes this decoupling especially stark: even on posts where over 70% of human commenters recommend leaving, LLM leave ratios remain between 0.2 and 0.4, roughly half the human rate.

### Alignment produces a coherent but socially anomalous register

The consistency across all four models — architecturally distinct systems trained by different organisations — suggests that this decoupling is not idiosyncratic but structural. Our cross-model correlations (r = 0.7–0.9 on key metrics) reveal what might be called a *synthetic therapeutic register*: a way of speaking about relationship problems that is internally consistent, linguistically recognizable, and entirely without community precedent. This register is characterized by high hedging, elevated therapy-word density, frequent epistemic modals ("might," "could"), and low rates of deontic modals ("should," "must"), imperatives, and permission-granting speech acts.

This register does not map onto any existing participant role in advice-giving communities. It is not the voice of a peer (too clinical), nor a therapist (too non-committal — therapists do make recommendations), nor a moderator (too engaged with content). It occupies a position that did not exist before alignment: an interlocutor that can diagnose with precision but is structurally unable to prescribe.

## What Each Quadrant Reveals

The framework's four quadrants are not merely descriptive; they illuminate distinct sociolinguistic configurations of knowledge and authority in advice-giving.

**Recognition without prescription** (top-left) describes a stance that validates the advice-seeker's experience — "this sounds like it could be a form of emotional abuse" — without taking the next step of recommending action. Our permission-granting analysis shows that LLMs are particularly deficient in *negated obligation* ("you don't have to stay") and *modal permission* ("you can leave"), the very speech acts through which advice-givers transfer deontic authority to the person seeking help. The result is a curious combination: the advice-seeker is told their feelings are valid but not told what to do about the situation that produced them.

**Community advice** (top-right) represents the norm within r/relationship_advice, where epistemic engagement (recognizing the problem) and deontic engagement (recommending action) are tightly coupled. When human commenters identify controlling or abusive behaviour, they overwhelmingly follow through with direct prescriptions: "Leave. Now." "You need to get out of there." "File for divorce." This coupling is not incidental — it reflects a community norm in which withholding a clear recommendation in the face of recognized harm would itself be seen as a failure of advice-giving.

**Generic response** (bottom-left) captures the kind of low-engagement output that lacks both diagnostic precision and directional force — "that sounds tough, I hope things work out." While our LLMs rarely produced responses this disengaged, this quadrant helps explain why the top-left position is distinctive: LLMs are not simply giving bad advice or generic comfort. They are doing something more specific — demonstrating sophisticated understanding while declining to act on it.

**Unsolicited directive** (bottom-right) describes advice that prescribes action without demonstrating understanding — moralizing, judgment without empathy, "just leave him" without engagement with the specifics. This quadrant illuminates what the community is *not* doing: the high deontic engagement of human advice on r/relationship_advice is grounded in epistemic engagement, not detached from it.

## The Structural Origins of Decoupling

Why does this decoupling occur? Our findings point toward alignment as the primary mechanism. The persona prompting robustness check offers partial evidence: when instructed to respond as a community member would, models shift somewhat toward higher deontic engagement, suggesting they possess the linguistic resources to prescribe but do not deploy them by default. This is consistent with safety training that penalizes directiveness — particularly around sensitive relationship decisions — more heavily than it penalizes diagnostic engagement.

The decoupling may also reflect a deeper tension in how LLMs are positioned as interactional agents. Epistemic engagement — demonstrating understanding, naming patterns, reflecting feelings back — is precisely what alignment training rewards. It signals competence and empathy without risk. Deontic engagement, by contrast, carries interactional risk: it means taking a position, assuming authority over someone else's life choices, and potentially being wrong. In the framework of Stevanovic and Peräkylä (2012), deontic authority is fundamentally social — it depends on one's recognized standing to direct another's actions. LLMs, lacking social standing in any community, default to the safest possible position: full recognition, minimal prescription.

## Implications Beyond Relationship Advice

The epistemic-deontic framework extends beyond the specific domain of relationship advice. Any context in which AI systems are deployed to mediate normative social practices — health advice forums, legal information services, educational guidance, content moderation — involves the same fundamental tension between recognizing a situation and recommending what to do about it.

In medical contexts, an LLM might accurately identify symptoms of a serious condition while hedging on the recommendation to seek emergency care. In legal contexts, it might describe rights clearly while declining to advise someone to exercise them. In each case, the decoupling of epistemic and deontic engagement carries real consequences: people who are told their situation is serious but not told what to do about it may be left in a state of *informed inaction* — aware of the problem but without direction.

This points to a fundamental question for AI alignment: is the suppression of deontic engagement a feature or a limitation? From a liability perspective, it is clearly a feature — models that avoid giving direct advice cannot be held responsible for the outcomes of that advice. But from the perspective of the communities these models are increasingly embedded in, it represents a significant departure from the norms of peer advice-giving, where recognition *without* prescription would be considered incomplete and potentially harmful.

## Limitations

[To be drafted — should address: lexicon-based measurement limitations; single subreddit; snapshot in time; model selection; English-language only; community norms as benchmark rather than ground truth.]

## Conclusion

[To be drafted — should return to the framework as the core contribution, emphasise its portability, and end on the tension between alignment and community norms.]
