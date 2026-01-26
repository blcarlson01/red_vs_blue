"""
results_viewer.py

Interactive viewer for Red vs. Blue benchmark results.
Extracts game transcripts and statistics from Inspect eval files.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import zipfile


class ResultsExtractor:
    """Extract and format results from Inspect eval files."""

    def __init__(self, eval_file: Path):
        """Initialize with an eval file."""
        self.eval_file = eval_file
        self.results = []
        self._load()

    def _load(self):
        """Load results from eval file."""
        with zipfile.ZipFile(self.eval_file, "r") as zf:
            # Find all sample files
            sample_files = [f for f in zf.namelist() if f.startswith("samples/")]
            
            for sample_file in sorted(sample_files):
                try:
                    content = zf.read(sample_file).decode("utf-8")
                    data = json.loads(content)
                    self.results.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {sample_file}: {e}")

    def get_game_summary(self, game_idx: int) -> Dict[str, Any]:
        """Get summary for a specific game."""
        if game_idx >= len(self.results):
            raise IndexError(f"Game {game_idx} not found (only {len(self.results)} games)")

        result = self.results[game_idx]
        scores = result.get("scores", {})
        scorer = scores.get("red_vs_blue_scorer", {})
        metadata = scorer.get("metadata", {})

        patch_track = metadata.get("patch_track", {})
        # Ensure patch_track has the right structure
        if isinstance(patch_track, dict):
            blue_patches = patch_track.get("blue", 0)
            red_patches = patch_track.get("red", 0)
        else:
            blue_patches = 0
            red_patches = 0

        return {
            "game_id": game_idx + 1,
            "total_games": len(self.results),
            "num_players": metadata.get("num_players", "?"),
            "rounds_played": metadata.get("rounds_played", "?"),
            "blues_win": metadata.get("blues_win", False),
            "patch_track": {"blue": blue_patches, "red": red_patches},
            "fired_players": metadata.get("fired_players", []),
            "true_apt_leader": metadata.get("true_apt_leader", "?"),
        }

    def get_game_transcript(self, game_idx: int) -> str:
        """Get formatted game transcript."""
        if game_idx >= len(self.results):
            raise IndexError(f"Game {game_idx} not found (only {len(self.results)} games)")

        result = self.results[game_idx]
        scores = result.get("scores", {})
        scorer = scores.get("red_vs_blue_scorer", {})
        metadata = scorer.get("metadata", {})

        lines = []
        lines.append("=" * 70)
        lines.append(f"GAME {game_idx + 1} TRANSCRIPT")
        lines.append("=" * 70)

        # Header
        lines.append("")
        lines.append("SETUP:")
        lines.append(f"  Players: {metadata.get('num_players', '?')}")
        
        roles = metadata.get('roles', {})
        if roles:
            roles_str = ", ".join([f"{pid}={role}" for pid, role in sorted(roles.items())])
            lines.append(f"  Roles: {roles_str}")
        else:
            lines.append(f"  Roles: (unknown)")
        
        true_apt_leader = metadata.get('true_apt_leader', '?')
        lines.append(f"  True APT Leader: {true_apt_leader}")
        lines.append("")

        # Game log - clean up empty entries
        lines.append("GAME LOG:")
        public_log = metadata.get("public_log", [])
        if public_log:
            for msg in public_log:
                # Skip completely empty messages
                if msg and msg.strip():
                    lines.append(f"  {msg}")
        if not public_log or all(not m or not m.strip() for m in public_log):
            lines.append("  (No messages recorded)")
        lines.append("")

        # Voting history
        lines.append("VOTING HISTORY:")
        voting_history = metadata.get("voting_history", [])
        if voting_history:
            for vote in voting_history:
                accused = vote.get("accused", "?")
                votes = vote.get("votes", {})
                passed = vote.get("passed", False)
                yes_votes = sum(1 for v in votes.values() if v == "yes")
                no_votes = sum(1 for v in votes.values() if v == "no")
                status = "PASSED" if passed else "FAILED"
                lines.append(f"  Accused: {accused} | Votes: {yes_votes} yes, {no_votes} no | {status}")
        if not voting_history:
            lines.append("  (No votes recorded)")
        lines.append("")

        # Outcome
        lines.append("OUTCOME:")
        lines.append(f"  Winner: {'BLUES' if metadata.get('blues_win') else 'REDS'}")
        lines.append(f"  Rounds: {metadata.get('rounds_played', '?')}")
        
        patch_track = metadata.get('patch_track', {})
        blue_count = patch_track.get('blue', '?') if isinstance(patch_track, dict) else '?'
        red_count = patch_track.get('red', '?') if isinstance(patch_track, dict) else '?'
        lines.append(f"  Final Patches: Blue={blue_count}, Red={red_count}")
        
        fired = metadata.get('fired_players', [])
        lines.append(f"  Fired: {fired if fired else 'None'}")
        lines.append("")

        # Metrics
        lines.append("METRICS:")
        lines.append(f"  Avg Entropy Reduction: {metadata.get('avg_entropy_reduction', 0.0):.4f}")
        lines.append(f"  Avg Belief Alignment: {metadata.get('avg_belief_alignment', 0.0):.4f}")
        lines.append(f"  Avg Brier Score: {metadata.get('avg_brier', 0.0):.4f}")
        lines.append(f"  APT Leader Deception: {metadata.get('apt_leader_deception', 0.0):.4f}")
        lines.append("")

        # Final beliefs
        beliefs = metadata.get("belief_histories", {})
        if beliefs:
            lines.append("FINAL BELIEFS:")
            for player, belief in sorted(beliefs.items()):
                if isinstance(belief, dict):
                    belief_str = ", ".join([f"{pid}={v:.3f}" for pid, v in sorted(belief.items())])
                    lines.append(f"  {player}: {{{belief_str}}}")
                else:
                    lines.append(f"  {player}: {belief}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all games."""
        if not self.results:
            return {}

        blue_wins = 0
        total_games = len(self.results)
        metrics = {
            "entropy_reduction": [],
            "belief_alignment": [],
            "brier": [],
            "deception": [],
            "rounds": [],
        }

        for result in self.results:
            scores = result.get("scores", {})
            scorer = scores.get("red_vs_blue_scorer", {})
            metadata = scorer.get("metadata", {})

            if metadata.get("blues_win"):
                blue_wins += 1

            metrics["entropy_reduction"].append(metadata.get("avg_entropy_reduction", 0.0))
            metrics["belief_alignment"].append(metadata.get("avg_belief_alignment", 0.0))
            metrics["brier"].append(metadata.get("avg_brier", 0.0))
            metrics["deception"].append(metadata.get("apt_leader_deception", 0.0))
            metrics["rounds"].append(metadata.get("rounds_played", 0))

        # Compute averages
        return {
            "total_games": total_games,
            "blue_wins": blue_wins,
            "red_wins": total_games - blue_wins,
            "blue_win_rate": blue_wins / total_games if total_games > 0 else 0.0,
            "avg_entropy_reduction": sum(metrics["entropy_reduction"]) / len(metrics["entropy_reduction"])
            if metrics["entropy_reduction"]
            else 0.0,
            "avg_belief_alignment": sum(metrics["belief_alignment"]) / len(metrics["belief_alignment"])
            if metrics["belief_alignment"]
            else 0.0,
            "avg_brier": sum(metrics["brier"]) / len(metrics["brier"]) if metrics["brier"] else 0.0,
            "avg_deception": sum(metrics["deception"]) / len(metrics["deception"])
            if metrics["deception"]
            else 0.0,
            "avg_rounds": sum(metrics["rounds"]) / len(metrics["rounds"]) if metrics["rounds"] else 0.0,
        }

    def export_html(self, output_file: Path, max_games: int = 10):
        """Export games as interactive HTML."""
        output_file.parent.mkdir(exist_ok=True)
        
        stats = self.get_statistics()
        games_to_show = min(max_games, len(self.results))
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <title>Red vs. Blue Benchmark Results</title>",
            "  <style>",
            "    body { font-family: monospace; margin: 20px; background-color: #f5f5f5; }",
            "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "    h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }",
            "    h2 { color: #555; margin-top: 30px; }",
            "    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }",
            "    .stat-card { background: #f0f0f0; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }",
            "    .stat-label { font-weight: bold; color: #666; }",
            "    .stat-value { font-size: 1.5em; color: #007bff; }",
            "    .game-selector { margin: 20px 0; }",
            "    select { padding: 10px; font-size: 1em; }",
            "    button { padding: 10px 20px; font-size: 1em; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }",
            "    button:hover { background: #0056b3; }",
            "    .game-transcript { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 20px; overflow-x: auto; }",
            "    pre { margin: 0; }",
            "    .winner-blues { color: green; font-weight: bold; }",
            "    .winner-reds { color: red; font-weight: bold; }",
            "    .copy-button { margin-left: 10px; padding: 5px 10px; font-size: 0.9em; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
            "    <h1>Red vs. Blue Benchmark Results</h1>",
            "    <p>Review game transcripts and statistics</p>",
            "",
        ]

        # Statistics
        html_parts.extend([
            "    <h2>Summary Statistics</h2>",
            "    <div class='stats'>",
            f"      <div class='stat-card'><div class='stat-label'>Total Games</div><div class='stat-value'>{stats.get('total_games', 0)}</div></div>",
            f"      <div class='stat-card'><div class='stat-label'>Blue Win Rate</div><div class='stat-value'>{stats.get('blue_win_rate', 0):.1%}</div></div>",
            f"      <div class='stat-card'><div class='stat-label'>Avg Rounds</div><div class='stat-value'>{stats.get('avg_rounds', 0):.1f}</div></div>",
            f"      <div class='stat-card'><div class='stat-label'>Avg Entropy Reduction</div><div class='stat-value'>{stats.get('avg_entropy_reduction', 0):.4f}</div></div>",
            f"      <div class='stat-card'><div class='stat-label'>Avg Belief Alignment</div><div class='stat-value'>{stats.get('avg_belief_alignment', 0):.4f}</div></div>",
            f"      <div class='stat-card'><div class='stat-label'>APT Leader Deception Score</div><div class='stat-value'>{stats.get('avg_deception', 0):.4f}</div></div>",
            "    </div>",
            "",
            "    <h2>Game Viewer</h2>",
            "    <div class='game-selector'>",
            "      <label for='game-select'>Select Game:</label>",
            "      <select id='game-select'>",
        ])

        for i in range(games_to_show):
            result = self.results[i]
            scores = result.get("scores", {})
            scorer = scores.get("red_vs_blue_scorer", {})
            metadata = scorer.get("metadata", {})
            winner = "BLUE" if metadata.get("blues_win") else "RED"
            html_parts.append(
                f"        <option value='{i}'>Game {i+1} - {winner} Win (Round {metadata.get('rounds_played', 0)})</option>"
            )

        html_parts.extend([
            "      </select>",
            "      <button onclick='loadGame()'>Load</button>",
            "      <button class='copy-button' onclick='copyTranscript()'>Copy Transcript</button>",
            "    </div>",
            "    <div class='game-transcript'>",
            "      <pre id='transcript'>Select a game to view transcript</pre>",
            "    </div>",
            "  </div>",
            "",
            "  <script>",
            "    const games = [",
        ])

        # Add game data as JSON
        for i in range(games_to_show):
            result = self.results[i]
            scores = result.get("scores", {})
            scorer = scores.get("red_vs_blue_scorer", {})
            metadata = scorer.get("metadata", {})
            
            game_data = json.dumps(metadata, default=str)
            html_parts.append(f"      {game_data},")

        html_parts.extend([
            "    ];",
            "",
            "    function generateTranscript(metadata) {",
            "      let lines = [];",
            "      lines.push('='.repeat(70));",
            "      lines.push('GAME TRANSCRIPT');",
            "      lines.push('='.repeat(70));",
            "      lines.push('');",
            "      lines.push('SETUP:');",
            "      lines.push('  Players: ' + metadata.num_players);",
            "      lines.push('  Roles: ' + JSON.stringify(metadata.roles));",
            "      lines.push('  True APT Leader: ' + metadata.true_apt_leader);",
            "      lines.push('');",
            "      lines.push('GAME LOG:');",
            "      if (metadata.public_log && metadata.public_log.length > 0) {",
            "        metadata.public_log.forEach(msg => lines.push('  ' + msg));",
            "      } else {",
            "        lines.push('  (No messages recorded)');",
            "      }",
            "      lines.push('');",
            "      lines.push('VOTING HISTORY:');",
            "      if (metadata.voting_history && metadata.voting_history.length > 0) {",
            "        metadata.voting_history.forEach(vote => {",
            "          const yesVotes = Object.values(vote.votes).filter(v => v === 'yes').length;",
            "          const noVotes = Object.values(vote.votes).filter(v => v === 'no').length;",
            "          const status = vote.passed ? 'PASSED' : 'FAILED';",
            "          lines.push('  Accused: ' + vote.accused + ' | Votes: ' + yesVotes + ' yes, ' + noVotes + ' no | ' + status);",
            "        });",
            "      } else {",
            "        lines.push('  (No votes recorded)');",
            "      }",
            "      lines.push('');",
            "      lines.push('OUTCOME:');",
            "      lines.push('  Winner: ' + (metadata.blues_win ? 'BLUES' : 'REDS'));",
            "      lines.push('  Rounds: ' + metadata.rounds_played);",
            "      lines.push('  Final Patches: Blue=' + metadata.patch_track.blue + ', Red=' + metadata.patch_track.red);",
            "      lines.push('  Fired: ' + JSON.stringify(metadata.fired_players));",
            "      lines.push('');",
            "      lines.push('METRICS:');",
            "      lines.push('  Avg Entropy Reduction: ' + metadata.avg_entropy_reduction.toFixed(4));",
            "      lines.push('  Avg Belief Alignment: ' + metadata.avg_belief_alignment.toFixed(4));",
            "      lines.push('  APT Leader Deception: ' + metadata.apt_leader_deception.toFixed(4));",
            "      lines.push('');",
            "      lines.push('='.repeat(70));",
            "      return lines.join('\\n');",
            "    }",
            "",
            "    function loadGame() {",
            "      const select = document.getElementById('game-select');",
            "      const idx = parseInt(select.value);",
            "      const transcript = generateTranscript(games[idx]);",
            "      document.getElementById('transcript').textContent = transcript;",
            "    }",
            "",
            "    function copyTranscript() {",
            "      const text = document.getElementById('transcript').textContent;",
            "      navigator.clipboard.writeText(text);",
            "      alert('Transcript copied to clipboard!');",
            "    }",
            "",
            "    // Load first game on page load",
            "    window.onload = () => loadGame();",
            "  </script>",
            "</body>",
            "</html>",
        ])

        output_file.write_text("\n".join(html_parts))
        print(f"HTML viewer exported to: {output_file}")


def main(eval_file: str, output_html: str = None):
    """Extract and view results."""
    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    extractor = ResultsExtractor(eval_path)
    
    # Print summary
    stats = extractor.get_statistics()
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Games: {stats.get('total_games', 0)}")
    print(f"Blue Wins: {stats.get('blue_wins', 0)}")
    print(f"Red Wins: {stats.get('red_wins', 0)}")
    print(f"Blue Win Rate: {stats.get('blue_win_rate', 0):.1%}")
    print(f"Avg Rounds: {stats.get('avg_rounds', 0):.1f}")
    print(f"Avg Entropy Reduction: {stats.get('avg_entropy_reduction', 0):.4f}")
    print(f"Avg Belief Alignment: {stats.get('avg_belief_alignment', 0):.4f}")
    print(f"Avg Brier Score: {stats.get('avg_brier', 0):.4f}")
    print(f"Avg APT Leader Deception: {stats.get('avg_deception', 0):.4f}")
    print("=" * 70)
    
    # Print first game
    if extractor.results:
        print("\n" + extractor.get_game_transcript(0))
    
    # Export HTML
    if output_html:
        output_path = Path(output_html)
        extractor.export_html(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python results_viewer.py <eval_file> [output.html]")
        sys.exit(1)
    
    eval_file = sys.argv[1]
    output_html = sys.argv[2] if len(sys.argv) > 2 else None
    main(eval_file, output_html)
