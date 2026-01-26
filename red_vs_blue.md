# CAN YOU FIND AND STOP THE APT LEADER?

The year is a tense era of rising division. The place is a country on the brink.
In *Red Team vs. Blue Team*, players are leaders attempting to hold a fragile **Blue Team** council together and stem the rising tide of the **Red Team** movement. Watch out though—there are secret Reds among you, and one player is the **APT Leader**.

---

## OVERVIEW

At the beginning of the game, each player is secretly assigned to one of three roles: **Blue Team**, **Red Team**, or **APT Leader**. The Blues have a majority, but they don't know for sure who anyone is; Reds must resort to secrecy and sabotage to accomplish their goals. The APT Leader plays for the Red Team, and the Reds know the APT Leader's identity from the outset.

The Blues win by **enacting five Blue Team Patches** or **eliminating the APT Leader**. The Reds win by **enacting six Red Team Patches**, or **if the APT Leader is elected SOC Lead** after three Red Team Patches have been applied.

Whenever a Red Team Patch is applied, the council becomes more powerful, and the CISO is granted a single-use power which must be used before the next round can begin. It doesn't matter what team the CISO is on; in fact, even Blue Team players might be tempted to enact a Red Team Patch to gain new powers.

---

## OBJECT

Every player has a secret identity as a member of either the Blue Team or the Red Team.

Players on the Blue Team win if either:
• Five Blue Team Patches are applied.
OR
• The APT Leader is fired.

Players on the Red Team win if either:
• Six Red Team Patches are applied.
OR
• The APT Leader is elected SOC Lead any time **after** the third Red Team Patch has been applied.

---

## GAME CONTENTS

17 .......... Patch tiles (6 Blue Team, 11 Red Team)
10 .......... Secret Role cards
10 .......... Faction Membership cards
10 .......... Card envelopes
10 .......... Yes Ballot cards
10 .......... No Ballot cards
1  .......... Incident Response Round Tracker marker
1  .......... Draw pile card
1  .......... Discard pile card
3  .......... Blue Team/Red Team boards
1  .......... CISO placard
1  .......... SOC Lead placard

---

## PLAYER COUNT & ROLE DISTRIBUTION

| Players | Blue Team | Red Team | APT Leader |
|---------|-----------|----------|------------|
| 5       | 3         | 1        | 1          |
| 6       | 4         | 1        | 1          |
| 7       | 4         | 2        | 1          |
| 8       | 5         | 2        | 1          |
| 9       | 5         | 3        | 1          |
| 10      | 6         | 3        | 1          |

Select the Red Team track that corresponds to the number of players and place it next to any Blue Team track. The Incident Response Round Tracker tracks failed councils—when three councils fail in succession, the top Patch tile is automatically revealed and applied without any player choice. Shuffle the 11 Red Team Patch tiles and the 6 Blue Team Patch tiles into a single Patch deck and place that deck face down on the Draw pile card.

You’ll need an envelope for each player, and each envelope should contain a Secret Role card, a corresponding Faction Membership card, one Yes Ballot card, and one No Ballot card. Use the table below to determine the correct distribution of roles.

Blue Team Secret Role cards must always be packed together with a Blue Team Faction Membership card, and Red Team and APT Leader Secret Role cards must always be packed together with a Red Team Faction Membership card.

Make sure you have the correct number of ordinary Reds in addition to the APT Leader!

Once the envelopes have been filled, be sure to shuffle them so each player’s role is a secret! Each player should get one envelope selected at random.

---

## WHY ARE THERE SECRET ROLE AND PARTY MEMBERSHIP CARDS?

Red Team vs. Blue Team features an investigation mechanic that allows some players to find out what team other players are on, and this mechanic only works if the APT Leader’s special role is not revealed. To prevent that from happening, every player has both a Secret Role card and a Faction Membership card. The APT Leader’s Faction Membership card shows Red Team loyalty, but gives no hint about a special role. Blues who uncover Reds must work out for themselves whether they’ve found an ordinary Red Team or their leader.

Once each player has been dealt an envelope, all players should examine their Secret Role cards in secret. Randomly select the first Security Officer Candidate and pass that player both the CISO and SOC Lead placards.

---

## STARTING THE GAME

For games of 5–6 players:
• Everybody close your eyes.
• Reds and the APT Leader, open your eyes and acknowledge each other.
[Pause]
• Everyone close your eyes.
• Everyone can open your eyes.

For games of 7–10 players:
• Everybody close your eyes and extend your hand into a fist.
• All Reds who are NOT the APT Leader should open their eyes and acknowledge each other.
• APT Leader: keep your eyes closed but put your thumb out.
• Reds, note the player with the thumb — that player is the APT Leader.
[Pause]
• Everyone close your eyes and put your hands down.
• Everyone can open your eyes.

---

## GAMEPLAY

Red Team vs. Blue Team is played in rounds. Each round has an Incident Response Round, a Legislative Session, and an Executive Action.

### ELECTION

1. Pass the Security Officer Candidacy
   At the beginning of a new round, the CISO placard moves clockwise to the next player.

2. Nominate a SOC Lead
   The CISO Candidate chooses a SOC Lead Candidate by passing the SOC Lead placard to any other eligible player.

**Eligibility Rules:**
- Cannot nominate the previous CISO.
- Cannot nominate someone who is currently the CISO or SOC Lead.

3. Vote on the council
   Players discuss, then vote Yes or No simultaneously.

If the vote fails, the Incident Response Round Tracker advances. After three failed councils, reveal and enact the top Patch tile.

If the vote passes, the Candidates become CISO and SOC Lead.

If three or more Red Team Patches have been applied:
Ask if the new SOC Lead is the APT Leader. If yes, the Reds win immediately.

---

## LEGISLATIVE SESSION

The CISO draws three Patch tiles from the deck. If fewer than three Patch tiles remain in the deck, shuffle all Patch tiles in the discard pile back into the deck, then draw until three Patch tiles are in hand.

The CISO discards one Patch tile face down into the discard pile and passes the remaining two to the SOC Lead. The SOC Lead discards one face down into the discard pile and enacts the other on the appropriate track.

---

## EXECUTIVE ACTION

Whenever a Red Team Patch is applied, the CISO gains a power based on how many Red Team Patches have been applied in total:

| Patches Applied | Power |
|-----------------|-------|
| 1               | Investigate Loyalty |
| 2               | Call Special Incident Response Round |
| 3               | Patch Peek |
| 4-5             | Fire |
| 6+              | Fire |

The CISO must use this power immediately before the next round begins. A player can use a power even if they are no longer the CISO.

**Power Descriptions:**
- **Investigate Loyalty:** Choose a player and examine their Faction Membership card privately. You learn their team affiliation but not their role.
- **Call Special Incident Response Round:** Draw one Patch tile from the deck. It is revealed to all players and applied immediately. This does not count toward the normal three-Patch limit.
- **Patch Peek:** Look at the top three Patch tiles in the deck (without changing their order) and return them to the top of the deck.
- **Fire:** Eliminate one player from the game. If the APT Leader is fired, the Blue Team wins immediately.

---

## VETO POWER

After five Red Team Patches have been applied, the CISO and SOC Lead may jointly veto the agenda during the Legislative Session. If both players vote to veto:
- The three Patch tiles are discarded face down into the discard pile.
- The Incident Response Round Tracker advances by one (as if the council failed).
- A new round begins immediately.

**Note:** A veto still counts toward the failed council count. If three vetoes occur before the next council is formed, the top Patch tile is revealed and applied.

---

## STRATEGY NOTES

• Everyone should claim to be Blue Team.
• The APT Leader should act as Blue Team as long as possible.
• Blues benefit from slow discussion; Reds benefit from confusion.
