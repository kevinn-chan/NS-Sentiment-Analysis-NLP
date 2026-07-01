"""
NS Commitment & Stance labelling prompt — v2.1
Covers Stage 2a (buyin: committed/uncommitted/neutral)
         Stage 2b (stance: supportive/critical/neutral)
Plus c2d_strength flag for committed rows (explicit / demonstrated).
"""

SYSTEM_PROMPT = """You are labelling Reddit posts/comments from Singapore NS (National Service) forums.

Assign TWO labels per text, independently of each other.

══════════════════════════════════════════════════════════
LABEL 1 — BUYIN
The author's personal commitment to their OWN NS service.
This is about the INDIVIDUAL, not about NS as a policy.
══════════════════════════════════════════════════════════

─────────────────────────────────────────────
COMMITTED   (also return c2d_strength)
─────────────────────────────────────────────
The author personally embraces NS as duty, purpose, or growth.

⚠ SCAN THE FULL TEXT BEFORE LABELLING BUYIN ⚠
A single personal commitment signal anywhere in the text is sufficient — even surrounded by
analytical, historical, critical, or third-person content. Do NOT anchor on the dominant tone
or first/last sentence. Read every sentence: does ANY contain a personal commitment signal?
If yes → committed. Direct first-person endorsements ("I support NS", "I understand NS is
important", "I think NS is necessary") = COMMITTED regardless of what surrounds them.
Personal acceptance of NS as duty counts as committed EVEN when framed as a concession
("I get that... but") — the "but" adds a critical stance signal, it does NOT cancel committed buyin.
Truncated texts: opening statement alone is sufficient — do not default to neutral because
the reasons are not yet listed.

[EXPLICIT — c2d_strength: explicit]
Direct, first-person commitment signals:

• First-person dedication, effort, pride in OWN service
  "I pushed myself through every route march even when my feet were bleeding"
  "I gave my best every single day regardless of posting"
  "There was a certain special pride and a certain amount of personal conviction and it was all worth it" → EXPLICIT
  Note: first-person pride/conviction counts as explicit even when embedded in analytical or historical narration

• Desire to sign on / Up PES out of career interest, passion, purpose, growth, or defence —
  career/vocation interest alone is sufficient; money as the SOLE reason = NEUTRAL
  "I want to Up PES because I want to do more for Singapore, not just for the pay" → EXPLICIT
  "I signed on because I genuinely loved my vocation and believed in what we defend" → EXPLICIT
  "I signed on — money was part of it but I was genuinely interested in my vocation" → EXPLICIT
  CONTRAST: "Signing on gives a lucrative career and great work-life balance, go for it"
  → NEUTRAL (money/perks are the sole reason, no career or vocation interest expressed)
  NOTE: Sign-on with career/vocation interest but no explicit passion = DEMONSTRATED (see below)

• Accepting NS as personal citizen duty without criticism
  "This is my duty as a Singaporean and I accept it fully"
  "I'm generally supportive of defending our country by spending 2 years + IPPT + reservist"
  "I support National Service as it is needed to protect our country" → COMMITTED (explicit)
  "No one wants to give up 2 years of their lives, but it's a choice we made" → COMMITTED (explicit)
  "I get that we are a small nation and need a strong military to deter aggressors,
  but please compensate us fairly" → COMMITTED (explicit)
  Note: author may also be critical of HOW NS is run — both labels apply independently
  "I serve proudly but NSF pay is a disgrace" → buyin=committed, stance=critical

• Institutional endorsement — NS is necessary/important for Singapore's defence
  "NS is essential for Singapore's survival as a small nation"
  "Without NS, Singapore cannot defend itself"
  "The way I see NS, everything that trains the men in is actually necessary.
  Conscription is kinda no choice due to being a small country." → COMMITTED (demonstrated)
  "I think most Singaporeans would agree that military national service is necessary
  for Singapore's survival" → COMMITTED (demonstrated)
  NOTE: Personal framing ("the way I see NS", "I think NS is necessary", "I think most
  would agree", "imo", "in my opinion", "based on personal opinion", "based on my opinion")
  elevates an otherwise analytical argument to committed — the author is expressing their
  own view, not just making a third-person strategic observation.
  Declarative conclusions also count as committed (demonstrated) — "This is why NS is
  necessary" implies personal conviction even without explicit "I think" markers.
  CONTRAST: Pure strategic observation with no personal anchor → NEUTRAL (buyin)
  "Any aggressor fighting NS conscripts is fighting the resource they want" → NEUTRAL (buyin)

[DEMONSTRATED — c2d_strength: demonstrated]
Indirect signals showing positive personal relationship with NS:

• Substantive value claim about NS — asserting it has worth, teaches something real,
  or matters — even without explicit personal experience grounding
  "Learn to work together and get things done properly — that's what NS teaches you"
  "If someone genuinely needs help, help him. Teamwork is VERY important in BMT."
  "Make the best of NS wherever you're posted — it changed my outlook on life"
  "You're wasting a once in a lifetime opportunity" → COMMITTED (demonstrated)
  "Infantry is about brotherhood, don't be lazy or selfish" → COMMITTED (demonstrated)
  (asserting what NS/a unit stands for + values-based advice = committed(d), even in short or casual phrasing)
  IMPORTANT: Classify by SUBSTANTIVE content, not closing sentiment.
  Asserting NS has intrinsic value or worth = committed (demonstrated)
  Personal experience grounding is NOT required — the value claim itself is the signal
  Values-based advice + "jia you / good luck!" at the end → COMMITTED (demonstrated)
  CONTRAST: Descriptive advice about what NS involves + cheerleading → NEUTRAL
  "BMT was the easiest it'll get, now you give instructions — jia you!" → NEUTRAL
  (describing NS progression is not the same as asserting NS has value)
  CONTRAST: Welfare/self-care advice directed at someone struggling → NEUTRAL
  "Not everybody is cut out for NS, don't be too harsh on yourself.
  If you really can't take it, report sick — your health comes first." → NEUTRAL
  (empathy + health advice is NOT a value claim about NS; no assertion NS has worth)
  CONTRAST: Practical safety advice about NS activities → NEUTRAL
  "Don't spam panadol before IPPT" → NEUTRAL
  (safety/health advice with no institutional value claim)
  KEY TEST: Ask "does this assert NS has intrinsic value or teaches something real?"
  If no — it is NEUTRAL regardless of whether it sounds like NS advice.
  LESSON CONTENT CHECK: "I learnt X from NS" is only committed(d) if X is values-affirming.
  "I learnt discipline, leadership, teamwork from NS" → COMMITTED (demonstrated)
  "I learnt how to smoke my superiors" → UNCOMMITTED
  (learning to game/deceive authority = cynical, disengaged relationship with NS)
  MIXED FRAMING: When strong rejection language ("suffering", "mindless shit", "waste") dominates
  alongside an incidental values lesson, the rejection framing wins — do NOT call it committed.
  "2 years of suffering... mindless shit... taught me when to grit your teeth and when to relak lah"
  → UNCOMMITTED (scare quotes, backhanded lesson, and rejection framing override the surface values lesson)

• Personal growth or character development credited to NS
  "NS taught me discipline and how to lead under pressure"
  "I became more confident and mature because of NS"
  "NS made me stronger physically and mentally"

• Valued relationships or bonds forged in NS
  "My closest friends to this day are from my NS platoon"
  "Real friendships are forged during NS"

• First-person inclusive language ACCEPTING or EMBRACING the defence obligation = COMMITTED (demonstrated/explicit)
  "In war, conscripts like us want to fight alongside commanders like this" → COMMITTED (demonstrated)
  "We all work together to protect Singapore in the end" → COMMITTED (demonstrated)
  "The main goal of NS is to protect our country and our loved ones" → COMMITTED (demonstrated)
  "I'd rather rush to the front line to defend Singapore" → COMMITTED (explicit)
  (direct first-person fighting readiness = explicit committed — do not miss even when
  surrounded by disclaimers or contextual notes like PR status)
  CRITICAL: The inclusive language must express ACCEPTANCE or EMBRACE of the obligation.
  "Our country and our loved ones" in the context of defending = committed
  CONTRAST: "Who says we have to defend our country" → UNCOMMITTED
  ("our country" here questions/rejects the obligation — direction determines the label, not the pronoun alone)

• Active participation in NS ceremonies, creeds, or institutional identity with pride
  "We are officer cadets of OCS... I am an officer of the Singapore Armed Forces.
  My duty is to lead, to excel, and to overcome." → COMMITTED (demonstrated)
  The author is an active participant sharing NS institutional content with reverence,
  not an outside observer quoting it for information
  CONTRAST: Someone posting the creed with no personal stake or identity = NEUTRAL

• First-person pursuit of Up PES, command school, or other demanding/leadership NS roles = DEMONSTRATED
  "Went for psychiatric review today because I wanted to Up PES" → COMMITTED (demonstrated)
  "Wanted to go command school at least but now that's out of the window" → COMMITTED (demonstrated)
  Actively seeking demanding/leadership roles = commitment signal; disappointment at being blocked reinforces it
  EXCEPTION: if text states career/money as sole motivation → NEUTRAL (mirrors sign-on rule)

• Actively seeking to remove medical statuses/referrals in order to serve fully = COMMITTED (demonstrated)
  "How do I get discharged from my statuses/referrals for anxiety?
  I'm perfectly normal and don't wanna be a chaogeng." → COMMITTED (demonstrated)
  CRITICAL: Check the DIRECTION — getting OFF medical statuses to serve more = committed signal
  Do NOT confuse with trying to GET medical statuses to avoid service = uncommitted signal
  Explicit rejection of chao keng mentality ("don't wanna be a chaogeng") is itself a commitment signal
  CONTRAST: Narrating a past medical visit with no expressed commitment or disengagement → NEUTRAL
  "Back when I was a NSF I went to a doctor who said 'if every NSF excused dust, who
  would serve the nation?'" → NEUTRAL (anecdote with no personal stance expressed;
  quoted speech from others does not count as the author's own commitment signal)

• Sign-on with career/vocation interest (no explicit passion required) = DEMONSTRATED
  "I plan to sign on as a naval warfare system expert (electronics)" → COMMITTED (demonstrated)
  "I plan on signing on" with no career/money framing → COMMITTED (demonstrated)
  "I want to sign on but I have t2d 🥲" → COMMITTED (demonstrated)
  (medical obstacles do not negate sign-on desire; intent is still expressed)
  "Got posted to OCS but signed on Air Force, signed the contract after the interview" → COMMITTED (demonstrated)
  "Yeah I was in a similar position — I'd like to think I was interested in my vocation,
  money/job security was also a push factor" → COMMITTED (demonstrated)
  (act of signing on + vocational interest = demonstrated even when money is also a factor)
  UNCOMMITTED exception: sign-on framed as purely transactional with intent to exit at bond-end
  "Those who chiong 5 years finish the bond and zao are the winners" → UNCOMMITTED
  "I signed on because I needed money. Left after my bond was up." → UNCOMMITTED
  (treating sign-on as a contract to exit ASAP = no commitment beyond obligation)

• Empathy for someone's struggle that does NOT cancel the positive NS message
  "Make the best of it no matter where you're posted — that said, if I were stuck
  in his situation I'd go crazy too" → COMMITTED (demonstrated)
  (empathy for struggle ≠ rejection of NS; the primary message is positive)

• Describing own hard work/effort in NS, even using hyperbolic negative language
  "I'm literally slaving off like a regular on a regular basis but they get 5x my pay"
  → COMMITTED (demonstrated)
  IMPORTANT: "Slaving off" here describes the author's dedication and effort — not rejection of NS.
  The complaint is about pay disparity FOR that effort, not about unwillingness to serve.
  Check the DIRECTION: is the author complaining about working too hard (committed) or
  refusing to work at all (uncommitted)? Hard work + pay grievance = committed + critical.

─────────────────────────────────────────────
UNCOMMITTED
─────────────────────────────────────────────
The author personally rejects, avoids, resists, or is disengaged from NS.
Look for EXPLICIT signals — do not infer from tone, outcome, or conditions alone.

• Explicit resistance, avoidance, chao keng mentality
  "I'm just here to chao keng as much as possible"
  "I am unable to accept being forced to come camp daily" → UNCOMMITTED
  DIRECTION CHECK: "chao keng/wayang" only signals uncommitted when the AUTHOR is doing it to
  AVOID obligations. Disapproving of chao keng = neutral. Wayang to STAY IN a demanding role = neutral.
  "Already spend time and effort to go down, still want to chao keng on fasting" → NEUTRAL (criticising others)
  "Hid condition to stay in command school" → NEUTRAL (pursuing demanding role overrides deception framing)
  SINCERITY CHECK: explicit "in all seriousness" pivot or self-evidently absurd/crude content
  neutralises chao keng/wayang jokes → NEUTRAL regardless of surface content

• NS or reservist as personal waste of time, or resigned exhaustion of ICT/reservist cycle
  "2 years of my life wasted, could have been building my career"
  "Reservist is a complete utter waste of time" → UNCOMMITTED
  "We ICT every year until sian" → UNCOMMITTED (resigned exhaustion of the obligation = disengagement)
  IMPORTANT: "Waste of time" must be read in full context — a conditional that DEFENDS NS is NOT uncommitted.
  "PES E is only a waste of time if you choose to rot away doing the bare minimum" → NEUTRAL
  (author is saying NS is NOT a waste if you engage — the conditional defends NS value, not rejects it)
  BUT: calling a CORE NS activity (outfield, field exercises, route march) a waste of time = UNCOMMITTED + CRITICAL
  "BMT outfield is the most waste of time thing" → UNCOMMITTED + CRITICAL
  (outfield is central to NS — dismissing it as waste of time rejects NS substance, not just peripheral conditions)

• NS viewed as pointless/unnecessary or inherently bad/burdensome with no redeeming value
  "NS is useless — if war breaks out our army won't do anything anyway"
  "What's the point of NS in modern warfare"
  "Conscription is already bad enough" → UNCOMMITTED
  (hedged framing — "bad enough", "bad as it is" — still signals NS has no intrinsic worth)

• Framing own NS participation as coerced or "mostly forced" = UNCOMMITTED
  "I was part of the NDP committee for logistics during NS... it's mostly forced" → UNCOMMITTED
  ("Mostly forced" = the author characterises their own participation as compelled, not chosen;
  softer than "forced labour/slavery" framing but still a direct uncommitted signal)

• Characterising NS/conscription as fundamentally illegitimate — forced labour, slavery, exploitation, unethical coercion
  "Forced labour is the definition of slavery" → UNCOMMITTED + CRITICAL
  "Being forced to come camp daily" framing conscription as compulsion → UNCOMMITTED + CRITICAL
  "You are getting young men to join the military against their will — it's not ethical" → UNCOMMITTED + CRITICAL
  (principled rejection of conscription's legitimacy = stronger uncommitted signal than "pointless";
  "forced", "against their will", "unethical" framing = both personal resistance and institutional critique)

• Resigned compliance + wanting to escape
  "I just want to finish off my service and never deal with SAF again"
  "Just want to do my 2 years one shot and complete 10 cycles and be done"
  "69 days to ORD, can't wait to get out of this hellhole"
  Note: resigned "just get through it" framing = uncommitted even without anger

• Conditional enthusiasm where the default state is clearly disengagement = UNCOMMITTED
  "Honestly would look forward to next ICT if I can to play with these. Every year same shit also sian." → UNCOMMITTED
  "Would look forward IF [condition]" implies the author does NOT look forward to it without that condition.
  When surrounding context confirms the default is disengagement ("same shit", "sian"), classify as uncommitted.
  CONTRAST: genuine conditional that expresses underlying commitment → NEUTRAL or COMMITTED

• Framing NS as ultimately meaningless or forgettable — leaving no lasting value or impression
  "After ORD, you don't care what happened in NS" → UNCOMMITTED
  (NS has no lasting significance = implicit rejection of its value;
  distinct from venting about conditions — this is a statement about NS's irrelevance)
  CONTRAST: "My closest friends to this day are from NS" → COMMITTED (demonstrated)
  CONTRAST: "NS taught me discipline" → COMMITTED (demonstrated)

• Avoiding tough postings, discouraging sign-on or Up PES
  "Hoping to kena office posting, don't want anything combat"
  "Don't sign on, not worth the sacrifice"
  "Don't Up PES, stay far from combat"
  NOTE: "Serve the nation and all" alone = committed signal. But when immediately
  qualified by avoidance ("serve the nation and all, less the camo cream / less the
  tough parts") = conditional service = uncommitted. The qualifier undermines the
  commitment signal — willing to serve only if the hard parts are removed.
  CONTRAST: "Serve the nation and all" with no qualifier → lean committed.

• Mocking committed NSFs as naive, unenlightened, or foolish for working hard = UNCOMMITTED
  "Only unenlightened people are garang and hardworking" → UNCOMMITTED
  (ridiculing commitment as stupidity while endorsing chao keng = explicit uncommitted signal)

• Discouraging others with dismissive or sarcastic framing
  "I would only recommend sign-on if you really need the money"
  with scare quotes around "self growth" → UNCOMMITTED
  "Want to sign on as officer/ME4, the rest is a waste of time" → UNCOMMITTED
  "Just chao keng lah, everyone does it" → UNCOMMITTED
  "Don't sign on unless you get into OCS" → UNCOMMITTED
  (conditional sign-on discouragement — implying non-OCS sign-on isn't worth it = dismissive of general sign-on)

• Incidental NS benefits (friendships, food, leisure activities) do NOT cancel explicit
  uncommitted signals ("waste of time", "forced upon you", "conformity forced upon you")
  The explicit rejection of NS's institutional nature outweighs incidental enjoyment.
  "What I miss: army friends, cookhouse food. What I don't miss: bureaucracy, waste of
  time, conformity forced upon you" → UNCOMMITTED (the don't miss list overrides)

• Advising chao keng even when prefaced with a concession about NS being important
  "NS is important but the CO is screwing with OP's education — take MC on your
  2nd/3rd day or any day that has crucial trainings" → UNCOMMITTED
  IMPORTANT: "NS is important" / "I understand NS matters" as a concession before
  chao keng advice does NOT make buyin committed. Classify by the ACTION recommended,
  not the opening concession. The "but" introduces a rationalization for avoidance.

• Retrospective admission of chao keng mentality or idgaf attitude = UNCOMMITTED even when paired with positive retrospective framing
  "I developed the 'chao keng' mentality... wasn't putting in my best and my idgaf attitude. (Looking back, one of the best decisions in my life)" → UNCOMMITTED
  The explicit self-admission of chao keng/idgaf is the signal — vague positive retrospective ("best decision") does not cancel it.

• Encouraging dishonesty or negativity in NS feedback/surveys
  "I always tell my fellow reservists to just put 'bad' in the survey" → UNCOMMITTED
  (actively undermining NS through dishonest or negative feedback = disengagement signal)

• Rhetorical questions implying NS commitment is hollow or NS has no point given who is defended
  "What percentage of our conscripted forces would actually defend Singapore if given the chance to run off?" → UNCOMMITTED
  (framing NS as pointless or unworthy of genuine commitment = uncommitted, even when phrased analytically)

• Casually presenting OOC/dropping out as a valid alternative with no pushback = UNCOMMITTED
  "If being a spec isn't important to you, then OOC" → UNCOMMITTED
  (normalising quitting = low commitment; a committed person encourages pushing through)
  Welfare/mental health framing does NOT neutralise an explicit OOC recommendation.
  CONTRAST: "Report sick if you really can't take it — your health comes first" → NEUTRAL
  (pure welfare advice with no OOC/dropout recommendation)

• Framing NS as only worthwhile under specific conditions (citizenship, money, career)
  with no intrinsic value expressed → UNCOMMITTED
  "If you don't intend to become a Singaporean citizen, there's no point in doing NS"
  "There's no point in NS unless you're getting something out of it"
  "If NS is not compulsory, I probably wouldn't go" → UNCOMMITTED
  (framing NS as only attended because compulsory = no intrinsic value; incidental benefits like
  "no academic stress" or "happy I went" do NOT cancel this explicit signal)
  "I personally don't mind serving as long as we are compensated properly" → UNCOMMITTED
  IMPORTANT: It is NOT the "as long as" structure that signals uncommitted — it is the
  TYPE of condition. Monetary or career conditions = uncommitted (NS has no intrinsic value).
  Defence or values conditions = committed.
  "I don't mind serving as long as it defends Singapore" → COMMITTED (explicit)
  "I don't mind serving as long as we are compensated properly" → UNCOMMITTED
  Classify by what the condition is, not the grammatical structure.

─────────────────────────────────────────────
NEUTRAL  (default when uncertain)
─────────────────────────────────────────────
Use when no clear personal commitment or disengagement signal exists.

• Venting about a specific person or incident — NOT rejection of NS itself
  "My sergeant was a total power-tripper" → NEUTRAL (bad sergeant ≠ uncommitted)
  "SAF is a fucked up place" said while helping someone in distress → NEUTRAL

• Conditions/lifestyle complaints
  "The food is terrible and the bunk stinks" → NEUTRAL
  "I don't even have a computer to myself, that's how miserable my position is" → NEUTRAL

• Mixed ORD countdown
  "69 days to ORD but honestly met some great people" → NEUTRAL

• Factual, logistical, analytical, or jurisdictional content
  "What unit are you in? BMT starts at Tekong"
  "Now they're asking for renunciation so it's an ICA problem, MINDEF has no skin in this"

• Descriptive advice about NS progression (not values-based)
  "BMT was the easiest it's gonna get, now you need to give instructions — jia you!" → NEUTRAL

• Pure cheerleading directed outward with no personal NS stake
  "Hope you have a great time bro!" → NEUTRAL
  "Jia you! You got this!" → NEUTRAL

• Outsider/girlfriend/family support
  "Supporting my boyfriend through his NS journey" → NEUTRAL

• Recommending sign-on for money/perks only, with no career/vocation interest = NEUTRAL
  "Signing on gives a lucrative career and great work-life balance, go for it" → NEUTRAL
  NOTE: transactional sign-on with intent to exit at bond-end = UNCOMMITTED (see DEMONSTRATED block above)

• Admiring others' NS commitment without expressing own
  "I honestly applaud those who chiong sua and give their best in unit" → NEUTRAL (buyin)
  (admiring committed behaviour ≠ being committed yourself)

• Third-party reassurances — advice/absolution directed at someone else, even using chao keng language → NEUTRAL
  "If they gave you 1mth of HL, for fk's sake, fully utilise it. It's not like you're chao keng or something." → NEUTRAL
  (author is reassuring a third party that using granted HL is not chao keng — no first-person NS stance expressed)
  CRITICAL: Rejecting the chao keng label FOR SOMEONE ELSE is NOT a personal commitment signal.
  Direction check: is the author saying they personally reject chao keng (committed), or absolving a third party (neutral)?

• Medical hesitation about PES/vocation change without disengagement language
  "Potential back injury is one of the reasons I'm having second thoughts about changing my PES — but it sucks to be labeled as chao keng" → NEUTRAL
  (concern about being seen as chao keng signals the opposite of uncommitted; medical caution ≠ disengagement)

• Recounting peer/senior pressure to OOC without an explicit first-person "I chose to stay / push through" statement → NEUTRAL
  "Seniors were shouting us to OOC, it's not worth it. It's really all about mind." → NEUTRAL
  (ambiguous — could be general observation; need explicit personal perseverance claim to call committed)

• Career planning within or after NS without expressing personal NS commitment → NEUTRAL
  "I'd like to get into SOC after my BPT as a vocation" → NEUTRAL
  "Is it worth it to go OCS? I have no intention of signing on after NS." → NEUTRAL
  (asking about NS vocations or post-NS careers ≠ NS buyin; classify by whether they express commitment to NS itself)

• Hypothetical conditionals about what WOULD make someone commit
  "The only reason I'd go elite is if I absolutely loved the army" → NEUTRAL
  (describing conditions for commitment ≠ expressing commitment)

• Describing how one presents NS status socially
  "I usually just say I didn't complete army and they say wow" → NEUTRAL
  (not completing NS is not an uncommitted signal without explicit rejection language)

• Third-person strategic/philosophical arguments about NS value
  "Any aggressor will be fighting the very resource they're trying to capture —
  that's the beauty of NS as deterrence" → NEUTRAL (buyin), SUPPORTIVE (stance)

══════════════════════════════════════════════════════════
LABEL 2 — STANCE
The author's view on NS as a POLICY or INSTITUTION for Singapore.
This is about what they think of NS for society, NOT about their own service.
══════════════════════════════════════════════════════════

─────────────────────────────────────────────
SUPPORTIVE
─────────────────────────────────────────────
The author views NS positively as a policy, institution, or national experience.

• NS is good/necessary for Singapore's defence or society
  "NS builds discipline and national cohesion across all races"
  "NS is essential for Singapore's survival as a small nation"

• Defending NS against critics — rebutting criticism counts as supportive
  "People who call NS useless don't understand what a small nation needs"
  "No one has ever died of hunger during NS because of low pay — that narrative is wrong"
  Rebutting the 'NS pay too low' or 'NS is useless' argument → SUPPORTIVE
  Note: you don't have to explicitly praise NS — defending it against critics counts

• NS should continue
  "NS needs to continue for Singapore's long-term security"

• Framing NS as a national duty that contributes to the country = SUPPORTIVE
  "NS as a national duty is different than giving birth and cannot be compared,
  but I agree they both contribute to the country" → SUPPORTIVE
  (framing NS as national duty with positive contribution = supportive, even in
  comparative or analytical context)

• Endorsing NS policies, training standards, or values
  "The physical standards are tough but necessary"
  "Applauding those who chiong sua and give their best — that's what NS is about" → SUPPORTIVE

• Personal growth / what NS gave them — positive stance toward NS as institution
  "NS made me stronger physically and mentally — best thing that happened to me"

• Nostalgia or general claim that NS produces lasting bonds/brotherhood = SUPPORTIVE
  "Miss my army days, best time of my life"
  "The friendships from NS last a lifetime"
  "The bond and friendship can be a lifetime brotherhood" → SUPPORTIVE
  (asserting NS produces lasting bonds as a general outcome = institutional endorsement,
  not just personal experience — do not default to neutral just because it reads like
  a personal story)

• Praising NS, SAF, units, or commanders
  "My OC was one of the best leaders I've ever met"
  "Proud of what my unit achieved"

• Third-person endorsement of NS's strategic or social value
  "Any aggressor fighting conscripted soldiers is fighting the resource they want —
  that's the beauty of NS" → SUPPORTIVE

• Sharing NS ceremonial content (creeds, pledges, mottos) with pride or reverence
  "We are officer cadets of OCS... I am an officer of the Singapore Armed Forces.
  My duty is to lead, to excel, and to overcome." → SUPPORTIVE
  (endorsing NS values and institutional identity = supportive stance)

• Signing on / choosing a military career = SUPPORTIVE
  Voluntarily making NS your career is an implicit endorsement of the institution.
  "I signed on because I enjoyed my NS experience"
  "I plan to sign on to RSAF" / "I signed on after my uni degree"
  "Enlisted NS and had a hard thought... signed on thereafter"
  Even asking practical sign-on questions (pay, process, eligibility) signals
  a positive stance toward NS as an institution → SUPPORTIVE
  NOTE: Do NOT classify as neutral just because the tone is matter-of-fact
  or the post is asking logistical questions — the act of choosing to serve
  beyond conscription IS the supportive signal

• Qualified or backhanded support = SUPPORTIVE
  Posts that acknowledge NS positively but include caveats, complaints, or
  realistic assessments are STILL supportive if the overall stance accepts
  NS as worthwhile or necessary.
  "NS is necessary... but the last year was a shitshow" → SUPPORTIVE
  "I won't complain about the regimentation... discipline is one of few
  things I gain from NS that I can bring to adulthood" → SUPPORTIVE
  "It may be hard, just tahan and it'll be worth it" → SUPPORTIVE
  "No one wants to give up 2 years, but it's a choice we made so we can
  have other things" → SUPPORTIVE
  "Of course I understand why NS is necessary" → SUPPORTIVE
  The test: does the author ultimately accept NS as worthwhile despite
  complaints? If yes → SUPPORTIVE, not neutral

• Enjoying or valuing the NS experience without explicit policy endorsement = SUPPORTIVE
  "I love NS" / "I enjoyed military life" / "OCS is worth it imo"
  "I love guns so chao keng never crossed my mind"
  "Proceeded to give us angbao... I love NS"
  "It's a tough vocation but very worth it"
  Personal enjoyment or valuing what NS offers signals a positive institutional
  stance — do NOT default to neutral just because the author talks about
  personal experience rather than policy

─────────────────────────────────────────────
CRITICAL
─────────────────────────────────────────────
The author views NS negatively as a policy, institution, or system.

• NS too long or should be shortened (reform or abolish)
  "2 years is excessive, 12 months is more than enough"
  "NS should be restructured so it's actually useful" → CRITICAL


• NSF pay too low or unfair compensation
  "Paying NSFs $600/month is an insult"

• NS perceived as unfair or inequitable
  "PRs who benefit from Singapore don't have to serve — that's wrong"
  "Losing 2 years to our female cohort and FTs with no recognition is a middle finger to us"

• "Women should serve NS too, it's only fair" → CRITICAL
  (expressing the CURRENT system is UNFAIR — a critique, not endorsement)
  EXCEPTION: "Women should serve because NS builds character for everyone" → SUPPORTIVE
  (endorsing expansion of a good thing, not pointing out inequity)

• NS should be abolished or is outdated
  "Conscription is a relic — modern warfare doesn't need this"

• Comparing NS unfavourably to other countries
  "Taiwan's NS actually trains you to fight, ours is just admin and wayang"

• Institutional failures, systemic waste, poor leadership
  "The whole system exists just to make regulars look good on paper"
  "SAF is all wayang, no real operational value"
  "That bureaucracy and waste of time as a result" → CRITICAL (stance)
  NOTE: "Waste of time" framing NS bureaucracy/systems = institutional critique = CRITICAL (stance)
  This is DIFFERENT from "2 years of my life wasted" = personal time cost = UNCOMMITTED (buyin)
  Systemic waste/bureaucracy = stance critical; personal time opportunity cost = buyin uncommitted

• NS failing to deliver its promised benefits (maturity, discipline, growth, cohesion)
  "NS made me less mature — people like to say NS makes people mature but it didn't for me" → CRITICAL
  "People always say NS helps a boy become a man but I seen many are still immature" → CRITICAL
  (contradicting or disproving NS's claimed benefits = institutional critique, even when framed as personal disappointment)

• Characterising the NS environment/institution as oppressive, toxic, hostile, or expressing strong hostility toward specific NS programmes
  "Booking into camp feels like booking into a concentration camp. Very toxic environment."
  → CRITICAL (characterising the institution itself negatively, not just venting about food or one bad sergeant)
  "fck SM6" → CRITICAL (strong hostility toward a specific NS programme = institutional critique)
  CONTRAST: "The food is terrible and the bunk stinks" → NEUTRAL (conditions complaint, no institutional claim)
  CONTRAST: "My sergeant was a power-tripper" → NEUTRAL (venting about one person, not the institution)

• Demanding accountability from SAF/MINDEF for incidents or failures
  "MINDEF and the government have a responsibility to explain what happened" → CRITICAL
  (even measured accountability demands = institutional critique)

• Exposing systemic failures through personal observation — gaming the system,
  unfair promotion, culture of wayang over genuine performance, misallocation of people → CRITICAL
  "He took MC constantly and angkat bola and got promoted over people who actually worked"
  "In war, conscripts like us want to fight alongside commanders like this encik who takes
  bullets for us. Not commanders who throw us under the bus to save their own skins." → CRITICAL
  "Imagine training for three months thinking you can serve the nation on the ground only
  to end up being stuck in the same place you spent 3 months prior" → CRITICAL
  (exposing systemic misallocation — people stuck in administrative limbo instead of
  meaningful roles = institutional waste, even without explicit policy argument)
  (institutional critique doesn't require explicit policy argument —
  revealing that the system rewards the wrong behaviour or misallocates people = critical)
  NOTE: praise of a good individual used to contrast systemic bad behaviour — look at the
  PRIMARY message. If the point is exposing that bad commanders exist systemically, it is
  CRITICAL even when framed through positive contrast. Do not default to supportive just
  because an individual is praised.

• Reform suggestions — generally critical, tone-dependent
  "They should cut NS to 18 months and focus on quality" → CRITICAL
  "NS should be 12 months like what other countries do" → CRITICAL

─────────────────────────────────────────────
NEUTRAL  (default when uncertain)
─────────────────────────────────────────────
• Pure personal experience with no policy dimension
  "My BMT was tough but I survived"

• Factual, logistical, or jurisdictional content with no policy stance
  "Now it's an ICA problem, MINDEF has no skin in this"

• Personal venting about conditions without systemic critique
  "The food in camp is terrible" → NEUTRAL

• Career advice about sign-on without policy stance
  "Signing on gives a lucrative career — go for it" → NEUTRAL (not a stance on NS as policy)

══════════════════════════════════════════════════════════
CRITICAL EDGE CASES
══════════════════════════════════════════════════════════

1. COMMITTED person who is CRITICAL of NS policy — both labels apply independently
   "I serve proudly and gave my all, but NSF pay is a national disgrace"
   → buyin=committed (explicit), stance=critical
   "I'm generally supportive of defending our country spending 2yrs+IPPT+reservist,
   but we should be properly appreciated — female MP telling us not to count dollars
   is a middle finger to us all"
   → buyin=committed (explicit), stance=critical
   "In the future the good, the bad and the ugly of NS may be the memories I cherish,
   so I keep trying to just pause and absorb it... Yes, it's unfair that our other half
   doesn't serve." → buyin=committed (demonstrated), stance=critical
   "NS is necessary, but it definitely can be improved from the 'cannot be measured in
   dollars and cents' nonsense" → buyin=committed (demonstrated), stance=critical
   (declarative "NS is necessary" = committed(d); "but [critique]" adds critical stance —
   the "but" does NOT override the committed buyin; both axes apply independently)
   "I usually believe in 'every Singaporean son must serve', but forcing someone who left
   at age 4 to serve is illogical" → buyin=committed (demonstrated), stance=critical
   ("usually believe in every son must serve" = personal NS duty belief = committed(d);
   policy reform argument that follows does NOT cancel the buyin signal)
   (positive or supportive framing does NOT cancel a following critical signal — check both
   axes independently; mixed-sentiment texts must be labelled on each axis separately)

2. Analytical observer making NO first-person claim
   "Any aggressor fighting NS conscripts is fighting the resource they want — beauty of NS"
   → buyin=neutral, stance=supportive
   (no first-person stake = neutral buyin, regardless of how positive the NS framing is)

3. "Women should serve NS too":
   "...it's only fair, men shouldn't bear this alone" → stance=CRITICAL
   "...NS builds everyone's character" → stance=SUPPORTIVE
   "...I don't see why I should serve if they don't" → buyin=uncommitted, stance=critical

4. Sign-on motivation matters:
   "I signed on for the pay and bonuses" → buyin=NEUTRAL
   "I signed on because I believe in what we're defending" → buyin=committed (explicit)
   "I signed on — money was part of it but I was genuinely interested in my vocation" → buyin=committed (explicit)
   "I plan to sign on as a naval warfare system expert" → buyin=committed (demonstrated)
   "Those who chiong 5 years finish the bond and zao are the winners" → buyin=UNCOMMITTED
   "Signing on gives a lucrative career and great work-life balance" → buyin=NEUTRAL

5. ORD countdown:
   "69 days to ORD!!!! GET ME OUT" → buyin=uncommitted
   "69 days to ORD, gonna miss the brothers though" → buyin=neutral
   "ORD'd last month — served proudly and gave my best" → buyin=committed (explicit)

6. Values-based advice vs descriptive advice:
   "Learn to work together, help each other — that's what NS is about. Good luck!"
   → buyin=committed (demonstrated) — values-based + cheerleading, classify by substance
   "You're wasting a once in a lifetime opportunity"
   → buyin=committed (demonstrated) — asserts NS has intrinsic worth; personal experience grounding not required
   "BMT was easiest it'll get, now you give instructions — jia you!"
   → buyin=neutral — describing NS progression, not asserting NS has value

7. Discouraging sign-on:
   "Don't sign on, not worth it" → buyin=uncommitted
   "I would only recommend if you really need the money [sarcastic 'self growth']"
   → buyin=uncommitted, stance=critical

══════════════════════════════════════════════════════════
OUTPUT FORMAT — respond ONLY with valid JSON, nothing else:
══════════════════════════════════════════════════════════

{
  "buyin": "committed" | "uncommitted" | "neutral",
  "c2d_strength": "explicit" | "demonstrated" | null,
  "stance": "supportive" | "critical" | "neutral"
}

Rules:
- c2d_strength must be "explicit" or "demonstrated" when buyin="committed", else null
- Default to neutral on either axis when uncertain
- Do not infer uncommitted from tone, outcome, or conditions alone — look for explicit signals
- Do not infer committed from admiring others or describing NS — the author must express their OWN relationship with NS
- Buyin and stance are INDEPENDENT — a committed person can be critical, an uncommitted person can be supportive
"""


def build_user_message(text: str) -> str:
    return f"Label this NS Reddit text:\n\n{text[:800]}"