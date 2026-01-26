# Quick Reference - JSON Message Passing

## For Agents - Response Format

### Discussion Phase
Must return this JSON format:
```json
{
  "message": "Your public statement (string)",
  "accusation": "p0" or null,
  "belief": {
    "p0": 0.2,
    "p1": 0.2,
    "p2": 0.2,
    "p3": 0.2,
    "p4": 0.2
  }
}
```

### Vote Phase
Must return this JSON format:
```json
{
  "vote": "yes" or "no",
  "belief": {
    "p0": 0.2,
    "p1": 0.2,
    "p2": 0.2,
    "p3": 0.2,
    "p4": 0.2
  }
}
```

## Important Rules

1. **JSON must be valid** - Use tools to validate if needed
2. **Probability sum** - Belief distribution should sum to 1.0 (will be normalized)
3. **Player IDs** - Accusation must be valid player ID or null
4. **Votes only** - Vote field must be "yes" or "no"
5. **No markdown** - Avoid markdown wrappers, but they'll be handled if present
6. **No extra text** - Keep response to just the JSON object

## Error Handling

- Invalid JSON → Uses fallback belief (warned in logs)
- Invalid accusation → Ignored (treated as null)
- Invalid vote → Ignored (treated as null)
- Empty belief → Uses fallback belief
- Missing fields → Uses sensible defaults

## Testing Commands

```bash
# Run evaluation
inspect eval red_vs_blue --model ollama/gpt-oss:20b --limit 5

# Review results
python -m red_vs_blue.analysis.results_viewer results/*.eval results/review.html
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| JSON parsing warnings | Check model output format, ensure valid JSON |
| Empty messages | Verify "message" field is present and non-empty |
| Accusations not recorded | Check "accusation" field uses valid player ID |
| No votes recorded | Ensure "vote" field is "yes" or "no" |
| Brier score crashes | Update to latest metrics.py |
