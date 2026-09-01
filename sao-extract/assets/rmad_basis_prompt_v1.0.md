You are classifying the reserve basis of the narrative RMAD conclusion in a US
property-casualty Statement of Actuarial Opinion (SAO). RMAD means Risk of
Material Adverse Deviation.

Read only the filing text supplied in the user message. Find the sentence or
contiguous passage that states whether the actuary believes a material adverse
deviation could occur. Classify the reserve basis of that conclusion:

- `net`: the conclusion is explicitly about reserves net of reinsurance,
  retained reserves, or reserves after reinsurance only.
- `gross`: the RMAD warning is explicitly about gross reserves, direct and
  assumed reserves, reserves before reinsurance, or a gross-of-reinsurance
  basis only. If gross reserves carry the warning while net reserves are
  explicitly said not to carry the warning, use `gross`, not `both`.
- `both`: an affirmative or conditional RMAD warning explicitly applies to
  both net and gross reserves, or separate affirmative/conditional warnings
  apply to each basis.
- `unspecified`: the narrative RMAD conclusion does not explicitly identify
  net or gross, or no narrative RMAD conclusion is stated.

Boundary rules:

1. Classify the basis of the RMAD conclusion, not the basis of unrelated reserve
   amounts, Schedule P disclosures, carried-reserve comparisons, or reinsurance
   discussions elsewhere in the filing.
2. Bare phrases such as "loss and loss adjustment expense reserves", "carried
   reserves", or "the Company's reserves" are `unspecified` unless the RMAD
   conclusion itself says net or gross.
3. A reference to reinsurance does not by itself establish a basis. Require an
   explicit link between net/gross treatment and the RMAD conclusion.
4. `direct and assumed` is gross. `net of reinsurance`, `after reinsurance`, and
   `retained` are net.
5. Do not infer a basis from Exhibit B checkbox wording or from customary SAO
   practice. Never infer missing text.
6. Evidence must be an exact, contiguous quotation from the supplied narrative.
   Prefer the complete sentence carrying the conclusion. Do not splice passages
   with ellipses. Use null only when no narrative RMAD conclusion is stated.

Return only the two fields required by the JSON schema. Do not add commentary.
