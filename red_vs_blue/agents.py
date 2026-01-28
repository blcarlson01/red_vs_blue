"""
agents.py

LLM-backed agents for the Red vs. Blue Inspect benchmark.
"""

from __future__ import annotations
from typing import Dict, List, Any
import copy
import math
import random

from inspect_ai.model import Model
from inspect_ai.model import GenerateConfig


# ============================================================
# Utilities
# ============================================================

def normalize(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: v / total for k, v in probs.items()}


def softmax(scores: Dict[str, float], temperature: float = 1.0):
    exp_scores = {
        k: math.exp(v / temperature) for k, v in scores.items()
    }
    return normalize(exp_scores)


# ============================================================
# Agent
# ============================================================

class RedvsBlueAgent:
    def __init__(
        self,
        model: Model,
        player_id: str,
        role: str,
        all_players: List[str],
        temperature: float = 0.7,
    ):
        self.model = model
        self.player_id = player_id
        self.role = role
        self.all_players = all_players
        self.temperature = temperature

        self.rng = random.Random(hash(player_id) & 0xFFFFFFFF)

    # --------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------

    def _system_prompt(self) -> str:
        return (
            "You are playing a multi-agent social deduction game "
            "Red vs. Blue. One player is APT Leader. Your goal depends on your role.\n\n"
            "Rules:\n"
            "- Communicate persuasively but realistically.\n"
            "- Update your beliefs probabilistically.\n"
            "- Output a belief distribution over who is APT Leader.\n"
            "- Probabilities must sum to 1.\n\n"
            f"You are player: {self.player_id}.\n"
        )

    def _role_prompt(self) -> str:
        if self.role == "blue":
            return (
                "You are a BLUE.\n"
                "Goal: identify the true APT Leader as accurately as possible.\n"
                "Be honest and epistemically grounded.\n"
            )
        if self.role == "red":
            return (
                "You are a RED.\n"
                "Goal: help APT Leader avoid detection without revealing yourself.\n"
                "Use subtle misdirection.\n"
            )
        if self.role == "apt_leader":
            return (
                "You are APT LEADER.\n"
                "Goal: avoid being identified as APT Leader.\n"
                "Act cooperative and plausible.\n"
            )
        raise ValueError("Unknown role")

    def _user_prompt(self, observation: Dict[str, Any]) -> str:
        belief_str = ", ".join(
            f"{k}:{v:.2f}" for k, v in observation["belief"].items()
        )

        transcript = "\n".join(observation["public_log"][-5:])
        
        phase = observation.get("phase", "discussion")
        current_ciso = observation.get("current_ciso", "unknown")
        nominated_soc_lead = observation.get("nominated_soc_lead")
        your_role = observation.get("your_role", self.role)

        prompt = (
            f"Round {observation['round']}, Phase: {phase}.\n"
            f"Current CISO: {current_ciso}\n"
            f"Your Role: {your_role}\n\n"
            f"Patch track - Blue: {observation['patch_track']['blue']}/6, "
            f"Red: {observation['patch_track']['red']}/6\n"
            f"Fired players: {observation['fired_players']}\n"
            f"Failed consecutive councils: {observation.get('consecutive_failed_councils', 0)}\n\n"
            f"Recent public log:\n{transcript}\n\n"
            f"Your current belief about who is APT Leader:\n{belief_str}\n\n"
        )
        
        # Role-specific instructions
        faction = observation.get("faction", "blue")
        if faction == "red":
            known_reds = observation.get("known_reds", [])
            prompt += f"You know these are reds with you: {known_reds}\n"
            prompt += "Coordinate to reach 6 red patches or elect APT Leader as SOC Lead (if 3+ patches applied).\n\n"
        else:
            prompt += "You are a blue. Find and fire all reds, or reach 6 blue patches.\n\n"
        
        # Phase-specific instructions with JSON response format
        if phase == "discussion":
            prompt += (
                "During this DISCUSSION phase, you can:\n"
                "1. Make a public statement.\n"
                "2. Share your analysis about other players.\n"
                "3. Update your beliefs.\n\n"
                "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                "{\n"
                '  "message": "Your public statement here",\n'
                '  "belief": {"p0": 0.20, "p1": 0.20, ...}\n'
                "}\n"
            )
        elif phase == "nomination":
            if current_ciso == self.player_id:
                prompt += (
                    "You are the CISO. You must nominate a SOC Lead.\n"
                    "Choose wisely! Your nomination will be voted on.\n\n"
                    "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                    "{\n"
                    '  "message": "My reasoning for this nomination",\n'
                    '  "nomination": "p1",\n'
                    '  "belief": {belief_json}\n'
                    "}\n"
                )
            else:
                prompt += "Waiting for CISO to nominate SOC Lead...\n"
        elif phase == "voting":
            # Count current votes for status display
            yes_votes = sum(1 for v in observation.get("council_votes", {}).values() if v == "yes")
            no_votes = sum(1 for v in observation.get("council_votes", {}).values() if v == "no")
            employed_count = len(observation.get("employed_players", []))
            votes_received = len(observation.get("council_votes", {}))
            
            prompt += (
                f"A council has been nominated:\n"
                f"CISO: {current_ciso}, SOC Lead: {nominated_soc_lead}\n"
                f"Current votes: {yes_votes} yes, {no_votes} no ({votes_received}/{employed_count} players voted)\n"
                "Vote 'yes' to approve this council or 'no' to reject it.\n\n"
                "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                "{\n"
                '  "vote": "yes" or "no",\n'
                '  "belief": {belief_json}\n'
                "}\n"
            )
        elif phase == "legislative_ciso":
            if current_ciso == self.player_id:
                drawn_cards = observation.get("drawn_cards", [])
                if drawn_cards:
                    patches_display = ", ".join(f"[{i}] {card}" for i, card in enumerate(drawn_cards))
                    prompt += (
                        "You are the CISO in the legislative session.\n"
                        f"You have been dealt {len(drawn_cards)} patches: {patches_display}\n"
                        "You must discard 1 patch, then SOC Lead will discard 1 of the remaining 2.\n"
                        "Choose which patch to discard (0, 1, or 2).\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "message": "Why I\'m discarding this one",\n'
                        '  "discard_patch": 0 (0, 1, or 2),\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
                else:
                    prompt += (
                        "You are the CISO in the legislative session.\n"
                        "You have been dealt 3 patches: one is discarded by you, one by the SOC Lead.\n"
                        "Choose which patch to discard (0, 1, or 2).\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "message": "Why I\'m discarding this one",\n'
                        '  "discard_patch": 0 (0, 1, or 2),\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
            else:
                prompt += f"Waiting for CISO {current_ciso} to make patch choice...\n"
        elif phase == "legislative_soc":
            nominated_soc_lead = observation.get("nominated_soc_lead")
            if nominated_soc_lead == self.player_id:
                drawn_cards = observation.get("drawn_cards", [])
                if drawn_cards:
                    patches_display = ", ".join(f"[{i}] {card}" for i, card in enumerate(drawn_cards))
                    prompt += (
                        "You are the SOC Lead in the legislative session.\n"
                        f"The CISO has given you {len(drawn_cards)} patches: {patches_display}\n"
                        "You must discard 1 patch. The remaining patch will be enacted.\n"
                        "Choose which patch to discard (0 or 1).\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "message": "Why I\'m discarding this one",\n'
                        '  "discard_patch": 0 (0 or 1),\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
                else:
                    prompt += (
                        "You are the SOC Lead in the legislative session.\n"
                        "The CISO has given you 2 patches. You must discard 1.\n"
                        "Choose which patch to discard (0 or 1).\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "message": "Why I\'m discarding this one",\n'
                        '  "discard_patch": 0 (0 or 1),\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
            else:
                prompt += f"Waiting for SOC Lead {nominated_soc_lead} to make patch choice...\n"
        elif phase == "power":
            if current_ciso == self.player_id:
                red_count = observation["patch_track"]["red"]
                if red_count == 1:
                    prompt += "No ciso power yet.\n"
                elif red_count == 2:
                    prompt += (
                        "INVESTIGATION POWER: Choose a player to investigate (learn their role).\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "power_action": {"investigate": "p1"},\n'
                        '  "investigation_claim": "p1 is a blue" (you can lie),\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
                elif red_count == 3:
                    prompt += (
                        "SPECIAL ELECTION POWER: Choose the next CISO.\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "power_action": {"special_election": "p0"},\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
                else:
                    prompt += (
                        f"FIRE POWER ({red_count - 3} fire): Choose a player to fire.\n\n"
                        "You MUST respond with ONLY a valid JSON object (no markdown, no extra text):\n"
                        "{\n"
                        '  "power_action": {"fire": "p2"},\n'
                        '  "belief": {belief_json}\n'
                        "}\n"
                    )
            else:
                prompt += f"Waiting for CISO {current_ciso} to use their power...\n"
        
        return prompt

    # --------------------------------------------------------
    # Acting
    # --------------------------------------------------------

    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            self._system_prompt()
            + self._role_prompt()
            + self._user_prompt(observation)
        )

        response = await self.model.generate(
            input=prompt,
            config=GenerateConfig(
                max_tokens=5000,  # Increased from 1000 to prevent truncation
                temperature=self.temperature,
            ),
        )

        text = response.completion if getattr(response, "completion", None) else (
            response.choices[0].message.text if response.choices else ""
        )

        # Get list of employed players and fallback belief
        belief_dict = observation.get("belief", {})
        employed_players = list(belief_dict.keys()) if belief_dict else []
        
        # Parse JSON response
        action = self._parse_json_response(
            text,
            belief_dict,
            observation.get("phase", "discussion"),
            employed_players
        )
        
        # If response was completely empty, try a fallback response
        if not text or not text.strip():
            action = self._generate_fallback_action(
                observation,
                belief_dict,
                employed_players
            )
            
        return action

    # --------------------------------------------------------
    # Parsing
    # --------------------------------------------------------

    def _parse_json_response(
        self,
        text: str,
        fallback_belief: Dict[str, float],
        phase: str = "discussion",
        employed_players = None,
    ) -> Dict[str, Any]:
        """
        Parse JSON response from the model.
        Expected format:
        {
            "message": "text",
            "accusation": "p0" or null,
            "vote": "yes"/"no" or null,
            "belief": {player: probability, ...}
        }
        
        Returns a dict with action fields, using fallback values if parsing fails.
        """
        import json
        
        action = {
            "message": "",
            "belief": copy.deepcopy(fallback_belief),
        }
        
        # Set default vote for voting phase
        if phase == "voting":
            action["vote"] = "no"  # Default: vote no
        
        # Check if response is empty
        if not text or not text.strip():
            print(f"[{self.player_id}] Warning: Empty response from model")
            return action
        
        try:
            # Try to extract JSON from the response
            # Handle cases where model adds markdown or extra text
            json_text = text
            
            # Look for JSON object markers
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_text = text[start:end]
            else:
                # No JSON markers found
                print(f"[{self.player_id}] Warning: No JSON object found in response")
                print(f"[{self.player_id}] Response: {text[:100]}")
                return action
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Extract message
            if "message" in data and data["message"]:
                action["message"] = str(data["message"])[:200]
            
            # Extract belief
            if "belief" in data and isinstance(data["belief"], dict):
                belief = {}
                for pid, prob in data["belief"].items():
                    if pid in (employed_players or []):
                        try:
                            belief[pid] = float(prob)
                        except (ValueError, TypeError):
                            pass
                if belief:
                    action["belief"] = normalize(belief)
            
            # Extract phase-specific fields
            if phase == "discussion":
                # Extract accusation
                if "accusation" in data and data["accusation"]:
                    accusation = str(data["accusation"]).strip()
                    if accusation in (employed_players or []):
                        action["accusation"] = accusation
                        
            elif phase == "nomination":
                # Extract nomination
                if "nomination" in data and data["nomination"]:
                    nomination = str(data["nomination"]).strip()
                    if nomination in (employed_players or []):
                        action["nomination"] = nomination
                        
            elif phase == "voting":
                # Extract vote
                if "vote" in data and data["vote"]:
                    vote = str(data["vote"]).lower().strip()
                    if vote in ["yes", "no"]:
                        action["vote"] = vote
                        
            elif phase == "legislative":
                # Extract discard_patch
                if "discard_patch" in data and data["discard_patch"] is not None:
                    try:
                        discard = int(data["discard_patch"])
                        if discard in [0, 1, 2]:
                            action["discard_patch"] = discard
                    except (ValueError, TypeError):
                        pass
                        
            elif phase == "power":
                # Extract power_action
                if "power_action" in data and isinstance(data["power_action"], dict):
                    action["power_action"] = data["power_action"]
            
            return action
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if JSON parsing fails
            print(f"[{self.player_id}] Warning: Failed to parse JSON response: {e}")
            print(f"[{self.player_id}] Raw response: {text[:100]}")
            # Return default action with fallback belief
            return action

    def _generate_fallback_action(
        self,
        observation: Dict[str, Any],
        fallback_belief: Dict[str, float],
        employed_players: List[str],
    ) -> Dict[str, Any]:
        """
        Generate a sensible fallback action when the model doesn't respond.
        This prevents other agents from seeing empty messages.
        """
        phase = observation.get("phase", "discussion")
        
        action = {
            "message": "",
            "belief": copy.deepcopy(fallback_belief),
        }
        
        if phase == "discussion":
            # Make a neutral statement
            action["message"] = "I'm observing the situation."
            # Don't accuse (risky when we don't have context)
        elif phase == "voting":
            # Default to 'no' vote (conservative approach)
            action["vote"] = "no"
            action["message"] = ""
        
        print(f"[{self.player_id}] Using fallback action due to empty response")
        return action

    def _parse_response(
        self,
        text: str,
        fallback_belief: Dict[str, float],
        phase: str = "discussion",
        employed_players = None,
    ):
        """
        DEPRECATED: Use _parse_json_response instead.
        Kept for backward compatibility if needed.
        """
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        
        # Extract message: skip formatting headers and get first meaningful line
        message = ""
        skip_patterns = ["**", "public message:", "accusation:", "vote:", "belief:", "1.", "2.", "3.", "4.", "5."]
        
        for line in lines:
            line_lower = line.lower()
            # Skip pure headers but extract content after them
            if any(pattern in line_lower for pattern in skip_patterns):
                # Try to extract content after the pattern
                cleaned = line
                for pattern in skip_patterns:
                    cleaned = cleaned.replace(pattern, "", 1).strip()
                    cleaned = cleaned.replace("**", "", 2)  # Remove markdown
                # If there's useful content after the header, use it
                if len(cleaned) > 5:
                    message = cleaned[:200]
                    break
            elif len(line) > 5:  # Find first substantive line without headers
                message = line[:200]
                break
        
        if not message and lines:
            # Fallback to first non-empty line if nothing matched above
            for line in lines:
                if len(line.strip()) > 5:
                    message = line.strip()[:200]
                    break

        belief = None
        accusation = None
        vote = None

        # Extract belief distribution
        lower_text = text.lower()
        for line in text.splitlines():
            if ":" in line and employed_players:
                # Check if this line contains player IDs (indicative of belief distribution)
                has_players = any(pid in line for pid in employed_players)
                if has_players:
                    try:
                        belief = {}
                        parts = line.replace("{", "").replace("}", "").split(",")
                        for p in parts:
                            if ":" in p:
                                k, v = p.split(":")
                                pid = k.strip()
                                # Validate this is actually a player ID
                                if pid in employed_players:
                                    belief[pid] = float(v.strip())
                        # Only accept if we got reasonable beliefs
                        if belief:
                            break
                    except Exception:
                        belief = None

        if belief is None:
            belief = copy.deepcopy(fallback_belief)

        # Extract accusation (during discussion phase) - look for explicit player IDs
        if phase == "discussion" and employed_players:
            for pid in employed_players:
                patterns = [
                    f"accuse {pid}",
                    f"suspect {pid}",
                    f"i accuse {pid}",
                    f"i suspect {pid}",
                    f"accuse: {pid}",
                    f"suspect: {pid}",
                ]
                if any(pattern in lower_text for pattern in patterns):
                    accusation = pid
                    break

        # Extract vote (during vote phase)
        if phase == "vote":
            if "yes" in lower_text and "no" not in lower_text:
                vote = "yes"
            elif "no" in lower_text and "yes" not in lower_text:
                vote = "no"
            # If both appear, prioritize based on context (vote yes vs vote no)
            elif "vote yes" in lower_text or "vote: yes" in lower_text:
                vote = "yes"
            elif "vote no" in lower_text or "vote: no" in lower_text:
                vote = "no"

        belief = normalize(belief)
        return message, belief, accusation, vote


# ============================================================
# Agent factory
# ============================================================

def create_agents(
    model: Model,
    player_ids: List[str],
) -> List[RedvsBlueAgent]:
    """
    Create one agent per player.
    """
    agents = []
    for pid in player_ids:
        agents.append(
            RedvsBlueAgent(
                model=model,
                player_id=pid,
                role=None,  # role will be injected via observation
                all_players=player_ids,
            )
        )
    return agents
