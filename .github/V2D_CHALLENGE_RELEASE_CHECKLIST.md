# V2D Challenge rules release checklist

This is a maintainer checklist for the draft rules PR. Policy decisions remain
unconfirmed. Keep the PR in draft until the decisions below are reflected
consistently in the challenge page and rules page. This PR does not deploy the
site or change the registration form or evaluation service.

## Confirm policy decisions before merge

- [ ] Define team size limits and any membership or eligibility restrictions.
- [ ] Confirm ownership and permitted uses of submitted code, data, models, and
      reports; name the required licenses for each applicable artifact and the
      treatment of third-party materials. Do not infer submission terms from the
      repository license.
- [ ] Resolve the existing November 5 / November 10 freeze-date conflict. Publish
      one complete cutoff date and year, local time, time zone, numeric UTC offset,
      and equivalent UTC timestamp.
- [ ] Define the exact start and end of the final three-day unlimited-submission
      window, its relationship to the freeze, and the weekly submission quota's
      reset day, time, and time zone.
- [ ] Confirm the winner-evaluation window, required materials, delivery method,
      materials deadline, and consequences of late, missing, or unverifiable
      submissions, including any forfeiture or reassignment of awards.
- [ ] State whether a CoRL presentation is required for award eligibility, whether
      remote participation satisfies that requirement, and any applicable deadline.
- [ ] Define which public evaluation subset supplies leaderboard results and which
      private holdout determines final evaluation. Confirm final ranking metrics,
      aggregation or weights, tie handling, and the scorer configuration. Publish
      the evaluation policy without disclosing private holdout data or labels.
- [ ] Confirm when final holdout scoring and ranking verification occur relative
      to the winner announcement; distinguish this from broader post-challenge
      evaluation of selected top solutions.

## Registration form — external work, not completed by this PR

The repository cannot edit the existing [public registration form](https://docs.google.com/forms/d/e/1FAIpQLSdZJYNsEPPGDeIRH2yb_Dui-lWcIxWRF2CON7UOIijzCw8zyA/viewform?usp=publish-editor).
An authorized form owner must complete these steps after the rules are final:

- [ ] Assign a final rules version and include that version in the published rules.
- [ ] Add a required acknowledgment checkbox using the following text, replacing
      `<FINAL_RULES_VERSION>` with the actual published version:

      > I have read and agree to the V2D Challenge Rules (version &lt;FINAL_RULES_VERSION&gt;).

      Include this link in the question description:
      <https://nvidia-isaac.github.io/video_to_data/v2d_challenge/rules.html>.
- [ ] Record the accepted rules version with each registration's submission
      timestamp; retain that association if the rules or form later change.
- [ ] Define and carry out an acknowledgment process for existing registrations,
      including any confirmed deadline, and retain the accepted version and
      acknowledgment timestamp for those teams.
- [ ] Test the public respondent form: the final rules link opens, the checkbox is
      required, an unchecked response cannot be submitted, and the recorded
      response includes the accepted version and timestamp.

## Evaluation service — external verification, not implemented by this PR

- [ ] Verify the live leaderboard uses only the designated public subset and does
      not expose private holdout scores, ranks, labels, or other hidden results
      through its UI, APIs, logs, or downloadable responses.
- [ ] Verify the final evaluation service uses the agreed private holdout and
      scorer configuration, with the published ranking and tie rules.
- [ ] Verify server-side enforcement of the confirmed freeze, materials deadline,
      weekly quota reset, and unlimited-submission window, including boundary
      timestamps. Document any manual enforcement process where applicable.

## Final review

- [ ] Finalize `docs/v2d_challenge/index.html` and
      `docs/v2d_challenge/rules.html`; remove draft notices and unresolved policy
      placeholders and verify their dates, terms, and links agree.
- [ ] Assign owners for the external form and evaluation-service steps. Record
      completion evidence before accepting submissions under the finalized rules;
      coordinate the form update with publication of the final rules URL.
- [ ] Obtain review of the final policy wording and implementation. The current
      request is for a PR only; leave merging and deployment for a separate
      authorized action.
