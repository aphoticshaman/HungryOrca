/**
 * EXPANDED CARD QUOTE DATABASE - DIVERSE & COMPREHENSIVE
 * 10-15 quotes per card, covering upright AND reversed meanings
 *
 * Sources: Biblical, Historical, Self-Help, BookTok, Movies, Philosophy, Pop Culture
 * Target: People who want to improve their lives while being entertained
 *
 * Format: upright/reversed arrays with 10-15 quotes each
 */

export const CARD_QUOTES_EXPANDED = {
  // ═══════════════════════════════════════════════════════════
  // MAJOR ARCANA - ALL 22 CARDS
  // ═══════════════════════════════════════════════════════════

  0: { // The Fool
    name: 'The Fool',
    upright: [
      { text: "Blessed are the meek, for they shall inherit the earth.", source: "Matthew 5:5 (Bible)" },
      { text: "A journey of a thousand miles begins with a single step.", source: "Lao Tzu" },
      { text: "Do not be too timid and squeamish about your actions. All life is an experiment.", source: "Ralph Waldo Emerson" },
      { text: "The person who risks nothing, does nothing, has nothing, is nothing.", source: "Leo Buscaglia" },
      { text: "Jump, and you will find out how to unfold your wings as you fall.", source: "Ray Bradbury" },
      { text: "Sometimes the only way to stay sane is to go a little crazy.", source: "Susanna Kaysen" },
      { text: "We're all mad here.", source: "Alice in Wonderland" },
      { text: "Everything you want is on the other side of fear.", source: "Jack Canfield" },
      { text: "The cave you fear to enter holds the treasure you seek.", source: "Joseph Campbell" },
      { text: "Life begins at the end of your comfort zone.", source: "Neale Donald Walsch" },
      { text: "Fortune favors the bold.", source: "Virgil" },
      { text: "What would you attempt to do if you knew you could not fail?", source: "Robert H. Schuller" },
    ],
    reversed: [
      { text: "Pride goes before destruction, and a haughty spirit before a fall.", source: "Proverbs 16:18 (Bible)" },
      { text: "Not all who wander are lost, but some definitely are.", source: "Cautionary Wisdom" },
      { text: "Recklessness is not courage. Impulse is not intuition.", source: "Self-Awareness Truth" },
      { text: "Leap and the net will appear—but maybe check if there's a net first.", source: "Practical Wisdom" },
      { text: "Naivety dressed up as optimism will still get you hurt.", source: "Hard Lessons" },
      { text: "You can be spontaneous without being stupid.", source: "Life Coach Wisdom" },
      { text: "The road to hell is paved with good intentions and zero follow-through.", source: "Accountability Truth" },
      { text: "Don't mistake chaos for freedom.", source: "Mindfulness Philosophy" },
      { text: "Running from your problems is still running.", source: "Therapy Insight" },
      { text: "You're not 'free-spirited,' you're avoiding commitment.", source: "Brutal Honesty" },
    ]
  },

  1: { // The Magician
    name: 'The Magician',
    upright: [
      { text: "I have seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion.", source: "Blade Runner" },
      { text: "The Watchers taught us the secrets of heaven: the movements of stars, the properties of roots, the working of metals. Forbidden knowledge that built civilizations.", source: "Book of Enoch (Apocrypha)" },
      { text: "Warlocks don't apologize for their power. We weaponize it.", source: "Masculine Witch Energy" },
      { text: "As above, so below. As within, so without. As the universe, so the soul.", source: "Hermes Trismegistus" },
      { text: "The ritual isn't the magic. Your will is. The candles and herbs are just theater for your monkey brain.", source: "Chaos Magic" },
      { text: "You are a way for the cosmos to know itself.", source: "Carl Sagan" },
      { text: "If the doors of perception were cleansed everything would appear to man as it is, infinite.", source: "William Blake / Aldous Huxley" },
      { text: "Every spell is a prayer. Every prayer is a spell. The universe doesn't care about your vocabulary.", source: "Practical Witchcraft" },
      { text: "Science is magic that works.", source: "Kurt Vonnegut" },
      { text: "SCIENCE RULES!", source: "Bill Nye" },
      { text: "The warlock knows: manifestation without action is just daydreaming with candles.", source: "Warlock Wisdom" },
      { text: "The only way to deal with an unfree world is to become so absolutely free that your very existence is an act of rebellion.", source: "Albert Camus" },
      { text: "Reality is that which, when you stop believing in it, doesn't go away.", source: "Philip K. Dick" },
      { text: "I am the master of my fate, I am the captain of my soul.", source: "Invictus" },
      { text: "The Kingdom of Heaven is within you.", source: "Luke 17:21 (Bible)" },
    ],
    reversed: [
      { text: "Pride goeth before destruction, and a haughty spirit before a fall.", source: "Proverbs 16:18 (Bible)" },
      { text: "All that glitters is not gold.", source: "Shakespeare" },
      { text: "The map is not the territory.", source: "Alfred Korzybski" },
      { text: "Cleverness is not wisdom.", source: "Euripides" },
      { text: "You can't con an honest man.", source: "The Sting" },
      { text: "Fake it till you make it only works if you eventually make it.", source: "Reality Check" },
      { text: "Manipulation dressed up as manifestation is still manipulation.", source: "Spiritual Bypass Truth" },
      { text: "All magic comes with a price.", source: "Once Upon a Time" },
      { text: "The trickster gets tricked in the end.", source: "Universal Law" },
      { text: "You're not a guru, you're just good at Google.", source: "Modern Charlatan" },
    ]
  },

  2: { // The High Priestess
    name: 'The High Priestess',
    upright: [
      { text: "Be still and know that I am God.", source: "Psalm 46:10 (Bible)" },
      { text: "She kept her magic quiet. Not because she was afraid, but because she knew power doesn't need to announce itself.", source: "Witch Wisdom" },
      { text: "The cards don't tell the future. They reveal what your soul already knows.", source: "Tarot Witch Truth" },
      { text: "Witches don't cast spells. We align ourselves with what already exists and whisper: 'become.'", source: "The Craft" },
      { text: "The psychedelic experience is a journey to new realms of consciousness.", source: "Timothy Leary" },
      { text: "She remembered things her soul knew before her body was born. That's the kind of witch you can't burn.", source: "Ancestral Magic" },
      { text: "In the depth of winter, I finally learned that within me there lay an invincible summer.", source: "Albert Camus" },
      { text: "Not all those who wander are lost.", source: "J.R.R. Tolkien" },
      { text: "The quieter you become, the more you can hear.", source: "Ram Dass" },
      { text: "My grandmother was a witch, my mother was a witch, I am a witch. The blood knows.", source: "Hereditary Witchcraft" },
      { text: "There is no reality except the one contained within us.", source: "Hermann Hesse" },
      { text: "The cure for pain is in the pain.", source: "Rumi" },
      { text: "Your visions will become clear only when you can look into your own heart.", source: "Carl Jung" },
      { text: "The most beautiful experience we can have is the mysterious.", source: "Albert Einstein" },
    ],
    reversed: [
      { text: "My people are destroyed for lack of knowledge.", source: "Hosea 4:6 (Bible)" },
      { text: "Secrets, secrets are no fun. Secrets, secrets hurt someone.", source: "Childhood Wisdom" },
      { text: "What you don't know CAN hurt you.", source: "Reality Truth" },
      { text: "Ignoring your intuition is expensive.", source: "Self-Trust Wisdom" },
      { text: "Silence isn't always golden—sometimes it's just yellow.", source: "Cowardice Call-Out" },
      { text: "You can't heal what you don't reveal.", source: "Recovery Truth" },
      { text: "The thing you're avoiding is the thing you need to face.", source: "Shadow Work" },
      { text: "Spiritual bypassing is still bypassing.", source: "Therapy Reality" },
      { text: "Not everything that is faced can be changed, but nothing can be changed until it is faced.", source: "James Baldwin" },
      { text: "Your secrets are safe with me. I wasn't even listening.", source: "Disconnect Truth" },
    ]
  },

  3: { // The Empress
    name: 'The Empress',
    upright: [
      { text: "I praise you, for I am fearfully and wonderfully made.", source: "Psalm 139:14 (Bible)" },
      { text: "She is clothed with strength and dignity, and she laughs without fear of the future.", source: "Proverbs 31:25 (Bible)" },
      { text: "Green witches know: you don't need a fancy altar. The earth IS the altar.", source: "Earth Magic" },
      { text: "Nature does not hurry, yet everything is accomplished.", source: "Lao Tzu" },
      { text: "The earth laughs in flowers.", source: "Ralph Waldo Emerson" },
      { text: "Plant seeds with intention. Water them with belief. Harvest with gratitude. That's the whole spell.", source: "Garden Witch" },
      { text: "She was a wildflower in a world of roses, and that was her power.", source: "Atticus" },
      { text: "You are not a drop in the ocean. You are the entire ocean in a drop.", source: "Rumi" },
      { text: "Luxury is in each detail.", source: "Hubert de Givenchy" },
      { text: "Kitchen witches, garden witches, hedge witches—we all know the same secret: magic is everywhere when you pay attention.", source: "Practical Witchcraft" },
      { text: "She remembered who she was and the game changed.", source: "Lalah Delia" },
      { text: "The future belongs to those who believe in the beauty of their dreams.", source: "Eleanor Roosevelt" },
      { text: "Let everything happen to you: beauty and terror. Just keep going. No feeling is final.", source: "Rainer Maria Rilke" },
    ],
    reversed: [
      { text: "Man cannot live on bread alone.", source: "Matthew 4:4 (Bible)" },
      { text: "You can't pour from an empty cup.", source: "Self-Care Wisdom" },
      { text: "Neglecting yourself while caring for everyone else isn't noble—it's martyr syndrome.", source: "Boundary Truth" },
      { text: "Creative block is usually fear dressed up as 'not the right time.'", source: "Artist Reality" },
      { text: "You're not selfish for having boundaries. You're just not a doormat.", source: "Self-Respect" },
      { text: "Mother Teresa you're not. And that's okay.", source: "Permission to Rest" },
      { text: "Burnout is what happens when you try to avoid being human for too long.", source: "Recovery Wisdom" },
      { text: "You can't nurture others if you're running on fumes.", source: "Caregiver Truth" },
      { text: "Rest is not a reward. It's a requirement.", source: "Wellness Reality" },
      { text: "The garden dies when the gardener forgets to water it.", source: "Self-Neglect Metaphor" },
    ]
  },

  4: { // The Emperor
    name: 'The Emperor',
    upright: [
      { text: "This is Sparta!", source: "300" },
      { text: "Tonight we dine in hell!", source: "300" },
      { text: "Give respect and honor to all to whom it is due.", source: "Romans 13:7 (Bible)" },
      { text: "The buck stops here.", source: "Harry S. Truman" },
      { text: "I never saw a wild thing sorry for itself.", source: "D.H. Lawrence" },
      { text: "A leader is one who knows the way, goes the way, and shows the way.", source: "John C. Maxwell" },
      { text: "In this world, you're either the butcher or the cattle.", source: "Mad Max" },
      { text: "Discipline is choosing between what you want now and what you want most.", source: "Abraham Lincoln" },
      { text: "Structure creates freedom.", source: "Jocko Willink" },
      { text: "You will never have a greater or lesser dominion than that over yourself.", source: "Leonardo da Vinci" },
      { text: "The price of greatness is responsibility.", source: "Winston Churchill" },
      { text: "Power is not given to you. You have to take it.", source: "Beyoncé" },
    ],
    reversed: [
      { text: "Absolute power corrupts absolutely.", source: "Lord Acton" },
      { text: "Any man who must say 'I am the king' is no true king.", source: "Game of Thrones" },
      { text: "Tyranny is the deliberate removal of nuance.", source: "Albert Maysles" },
      { text: "Control is an illusion. Influence is real.", source: "Leadership Wisdom" },
      { text: "You can't build an empire on a cracked foundation.", source: "Construction Metaphor" },
      { text: "The emperor has no clothes.", source: "Hans Christian Andersen" },
      { text: "Fear is not respect. Compliance is not loyalty.", source: "Authority Truth" },
      { text: "A title doesn't make you a leader. Your actions do.", source: "Leadership Reality" },
      { text: "Micromanaging is a symptom of distrust, and distrust is a symptom of incompetence.", source: "Management Truth" },
      { text: "The only thing necessary for the triumph of evil is for good men to do nothing.", source: "Edmund Burke" },
    ]
  },

  5: { // The Hierophant
    name: 'The Hierophant',
    upright: [
      { text: "Iron sharpens iron, and one man sharpens another.", source: "Proverbs 27:17 (Bible)" },
      { text: "When the student is ready, the teacher will appear.", source: "Buddhist Proverb" },
      { text: "Tradition is not the worship of ashes, but the preservation of fire.", source: "Gustav Mahler" },
      { text: "We stand on the shoulders of giants.", source: "Isaac Newton" },
      { text: "The teacher who is indeed wise does not bid you to enter the house of his wisdom but rather leads you to the threshold of your mind.", source: "Kahlil Gibran" },
      { text: "Education is the most powerful weapon which you can use to change the world.", source: "Nelson Mandela" },
      { text: "Real knowledge is to know the extent of one's ignorance.", source: "Confucius" },
      { text: "In learning you will teach, and in teaching you will learn.", source: "Phil Collins" },
      { text: "The beautiful thing about learning is that no one can take it away from you.", source: "B.B. King" },
      { text: "Study the past if you would define the future.", source: "Confucius" },
      { text: "Wisdom is not a product of schooling but of the lifelong attempt to acquire it.", source: "Albert Einstein" },
    ],
    reversed: [
      { text: "Beware of false prophets, who come to you in sheep's clothing.", source: "Matthew 7:15 (Bible)" },
      { text: "The institution is sound, but the people are corrupt.", source: "Systemic Failure" },
      { text: "Tradition for tradition's sake is just peer pressure from dead people.", source: "Modern Wisdom" },
      { text: "Don't follow the crowd—they're probably lost too.", source: "Independent Thought" },
      { text: "Blind faith is dangerous. Ask questions.", source: "Critical Thinking" },
      { text: "Just because everyone's doing it doesn't mean it's right.", source: "Moral Courage" },
      { text: "The system is rigged, but you don't have to play by their rules.", source: "Rebellion Energy" },
      { text: "Question authority. Think for yourself.", source: "Timothy Leary" },
      { text: "Conformity is the jailer of freedom and the enemy of growth.", source: "JFK" },
      { text: "We've always done it this way' is the enemy of progress.", source: "Innovation Truth" },
    ]
  },

  6: { // The Lovers
    name: 'The Lovers',
    upright: [
      { text: "Love one another as I have loved you.", source: "John 13:34 (Bible)" },
      { text: "He fell first. He fell harder. And when she finally fell? They both burned.", source: "Romance Aesthetic" },
      { text: "Surrender is not weakness when you choose who holds the leash.", source: "Power Exchange Philosophy" },
      { text: "There is beauty in yielding. There is power in submission chosen freely.", source: "Erotic Gothic Aesthetic" },
      { text: "To love and be loved is to feel the sun from both sides.", source: "David Viscott" },
      { text: "I am my beloved's and my beloved is mine.", source: "Song of Solomon 6:3 (Bible)" },
      { text: "The bond snapped into place and suddenly breathing without them felt impossible.", source: "Fated Mates Energy" },
      { text: "Mine. That's what his eyes said. That's what her soul answered.", source: "Possession Romance" },
      { text: "When you realize you want to spend the rest of your life with somebody, you want the rest of your life to start as soon as possible.", source: "When Harry Met Sally" },
      { text: "You should be kissed, and often, by someone who knows how.", source: "Gone with the Wind" },
      { text: "Love is or it ain't. Thin love ain't love at all.", source: "Toni Morrison" },
      { text: "I choose you. And I'll choose you over and over. Without pause, without doubt, in a heartbeat. I'll keep choosing you.", source: "Modern Vows" },
      { text: "The kind of love that ruins you for anyone else. That's the only kind worth having.", source: "BookTok Truth" },
      { text: "He looked at her like she was the answer to every question he'd ever had.", source: "Devotion Energy" },
      { text: "Two souls recognizing each other across lifetimes: 'Oh. It's you. It's always been you.'", source: "Soul Recognition" },
    ],
    reversed: [
      { text: "Do not be unequally yoked with unbelievers.", source: "2 Corinthians 6:14 (Bible)" },
      { text: "We accept the love we think we deserve.", source: "Perks of Being a Wallflower" },
      { text: "Loving someone who doesn't love you back is like hugging a cactus. The tighter you hold on, the more it hurts.", source: "Unrequited Love Truth" },
      { text: "You can't save people, you can only love them.", source: "Anaïs Nin" },
      { text: "The worst distance between two people is misunderstanding.", source: "Communication Wisdom" },
      { text: "Staying in a relationship for fear of being alone is emotional bankruptcy.", source: "Self-Worth Philosophy" },
      { text: "Red flags look like flags when you're wearing rose-colored glasses.", source: "BoJack Horseman Wisdom" },
      { text: "Love without respect is dangerous. Respect without love is cold. You need both.", source: "Relationship Truth" },
      { text: "Sometimes the love isn't wrong, the timing is.", source: "Modern Heartbreak" },
      { text: "You deserve someone who chooses you every single day, not just when it's convenient.", source: "Standards Reminder" },
      { text: "Don't set yourself on fire to keep someone else warm.", source: "Self-Preservation" },
      { text: "Chemistry without compatibility is a recipe for disaster.", source: "Dating Reality" },
    ]
  },

  7: { // The Chariot
    name: 'The Chariot',
    upright: [
      { text: "I must not fear. Fear is the mind-killer.", source: "Dune" },
      { text: "What matters most is how well you walk through the fire.", source: "Charles Bukowski" },
      { text: "Witness me!", source: "Mad Max: Fury Road" },
      { text: "It is not the mountain we conquer, but ourselves.", source: "Edmund Hillary" },
      { text: "The only way out is through.", source: "Robert Frost" },
      { text: "Do not pray for easy lives. Pray to be stronger men.", source: "JFK" },
      { text: "Victory has a hundred fathers, but defeat is an orphan.", source: "JFK" },
      { text: "I can do this all day.", source: "Captain America" },
      { text: "The impediment to action advances action. What stands in the way becomes the way.", source: "Marcus Aurelius" },
      { text: "If you're going through hell, keep going.", source: "Winston Churchill" },
      { text: "Control what you can. Confront what you cannot.", source: "Stoic Wisdom" },
    ],
    reversed: [
      { text: "Don't mistake motion for progress.", source: "Denzel Washington" },
      { text: "You can't drive forward while looking in the rearview mirror.", source: "Life Coach Truth" },
      { text: "Control is an illusion. Flow is reality.", source: "Surrender Wisdom" },
      { text: "Forcing it is not manifesting it.", source: "Spiritual Reality" },
      { text: "Sometimes the bravest thing you can do is stop.", source: "Rest Philosophy" },
      { text: "You're fighting the wrong battle.", source: "Strategic Truth" },
      { text: "Aggression without direction is just violence.", source: "Warrior Wisdom" },
      { text: "The hardest battles are the ones we fight with ourselves.", source: "Internal Conflict" },
      { text: "Let go or be dragged.", source: "Zen Proverb" },
      { text: "You can't win a race you're not supposed to be running.", source: "Purpose Alignment" },
    ]
  },

  8: { // Strength
    name: 'Strength',
    upright: [
      { text: "There were giants in the earth in those days; and also after that, when the sons of God came in unto the daughters of men.", source: "Genesis 6:4 (Nephilim, Bible)" },
      { text: "I am not afraid of storms, for I am learning how to sail my ship.", source: "Louisa May Alcott" },
      { text: "She's whiskey in a teacup—looks sweet, burns going down, leaves you wanting more.", source: "Strong FMC Energy" },
      { text: "The lion and the calf shall lie down together but the calf won't get much sleep.", source: "Woody Allen" },
      { text: "I am no bird; and no net ensnares me.", source: "Jane Eyre" },
      { text: "Soft heart. Sharp edges. Dangerous combination.", source: "Modern Heroine" },
      { text: "The most common way people give up their power is by thinking they don't have any.", source: "Alice Walker" },
      { text: "Do not mistake my kindness for weakness.", source: "Al Capone" },
      { text: "She wore her scars like wings—proof she had survived the fall and learned to fly.", source: "Survivor Energy" },
      { text: "Speak softly and carry a big stick.", source: "Theodore Roosevelt" },
      { text: "True strength is keeping everything together when everyone expects you to fall apart.", source: "Resilience Truth" },
      { text: "I have not failed. I've just found 10,000 ways that won't work.", source: "Thomas Edison" },
    ],
    reversed: [
      { text: "You're not required to set yourself on fire to keep other people warm.", source: "Boundary Setting" },
      { text: "Compassion fatigue is real. Rest is required.", source: "Caregiver Truth" },
      { text: "Being the strong one doesn't mean you can't break.", source: "Permission to Feel" },
      { text: "Martyrdom is manipulation dressed as sacrifice.", source: "Brutal Honesty" },
      { text: "You can't save everyone. And that's okay.", source: "Limits Acceptance" },
      { text: "Sometimes being strong means asking for help.", source: "Vulnerability Courage" },
      { text: "The weight you're carrying was never yours to bear.", source: "Release Permission" },
      { text: "Tired of being the strong friend? Say so.", source: "Communication Reality" },
      { text: "You're allowed to be both a masterpiece and a work in progress.", source: "Self-Compassion" },
    ]
  },

  9: { // The Hermit
    name: 'The Hermit',
    upright: [
      { text: "For in much wisdom is much grief, and he that increaseth knowledge increaseth sorrow.", source: "Ecclesiastes 1:18 (Bible)" },
      { text: "The solitary witch learns to trust their own counsel. No coven can give you what silence teaches.", source: "Solitary Practice" },
      { text: "Knowing yourself is the beginning of all wisdom.", source: "Aristotle" },
      { text: "I went to the woods because I wished to live deliberately.", source: "Thoreau" },
      { text: "The warlock in the tower, the witch in the woods—we seek the same thing: truth without witnesses.", source: "Hermit Witch" },
      { text: "In solitude, the mind gains strength and learns to lean upon itself.", source: "Laurence Sterne" },
      { text: "Silence is not empty. It is full of answers.", source: "Meditation Wisdom" },
      { text: "The cave you fear to enter holds the treasure you seek.", source: "Joseph Campbell" },
      { text: "Sometimes the strongest magic happens when you close the grimoire and just listen.", source: "Intuitive Witchcraft" },
      { text: "Solitude is where I place my chaos to rest and awaken my inner peace.", source: "Nikki Rowe" },
      { text: "Turn off your mind, relax, and float downstream.", source: "The Beatles" },
      { text: "Moksha comes through knowing thyself completely.", source: "Vedic Wisdom" },
      { text: "The quieter you become, the more you can hear.", source: "Ram Dass" },
    ],
    reversed: [
      { text: "Isolation is not the same as introspection.", source: "Mental Health Truth" },
      { text: "You can't find yourself in a cave you refuse to leave.", source: "Stuck Pattern" },
      { text: "Hermit mode has an expiration date. Check yours.", source: "Balance Reality" },
      { text: "Hiding from the world is not spiritual—it's avoidance.", source: "Spiritual Bypass" },
      { text: "Loneliness and solitude are not the same thing.", source: "Distinction Clarity" },
      { text: "You've been 'processing' for months. Time to participate.", source: "Action Call" },
      { text: "Connection is part of wholeness, not a distraction from it.", source: "Integration Truth" },
      { text: "The guru on the mountaintop still has to come down eventually.", source: "Reality Check" },
      { text: "Withdrawal is a symptom, not a solution.", source: "Depression Awareness" },
    ]
  },

  10: { // Wheel of Fortune
    name: 'Wheel of Fortune',
    upright: [
      { text: "To everything there is a season, and a time to every purpose under heaven.", source: "Ecclesiastes 3:1 (Bible)" },
      { text: "What has been will be again, what has been done will be done again; there is nothing new under the sun.", source: "Ecclesiastes 1:9 (Bible)" },
      { text: "This too shall pass.", source: "Persian Proverb" },
      { text: "The wheel weaves as the wheel wills.", source: "Wheel of Time" },
      { text: "Luck is what happens when preparation meets opportunity.", source: "Seneca" },
      { text: "Fortune favors the prepared mind.", source: "Louis Pasteur" },
      { text: "The universe is under no obligation to make sense to you.", source: "Neil deGrasse Tyson" },
      { text: "What goes around comes around. Karma's only a bitch if you are.", source: "Cosmic Justice" },
      { text: "Fate whispers to the warrior, 'You cannot withstand the storm.' The warrior whispers back, 'I am the storm.'", source: "Warrior Philosophy" },
      { text: "Accept what is, let go of what was, have faith in what will be.", source: "Sonia Ricotti" },
      { text: "Life is like riding a bicycle. To keep your balance, you must keep moving.", source: "Albert Einstein" },
      { text: "The only constant in life is change.", source: "Heraclitus" },
      { text: "When patterns are broken, new worlds emerge.", source: "Tuli Kupferberg" },
    ],
    reversed: [
      { text: "Bad luck is just good luck in disguise, waiting for perspective.", source: "Reframe Wisdom" },
      { text: "If you don't like where the wheel is taking you, get off and build your own cart.", source: "Self-Determination" },
      { text: "Resisting change is like holding your breath. You can do it, but not forever.", source: "Flow Reality" },
      { text: "The wheel can't turn if you're standing in the spokes.", source: "Obstacle Metaphor" },
      { text: "Sometimes you win, sometimes you learn.", source: "Growth Mindset" },
      { text: "Stuck in a cycle? The exit is where you entered.", source: "Pattern Break" },
      { text: "You can't control the wind, but you can adjust your sails.", source: "Adaptation Truth" },
      { text: "The wheel is turning, but you're not on it anymore.", source: "Left Behind Energy" },
      { text: "Clinging to good fortune prevents new fortune from arriving.", source: "Attachment Lesson" },
    ]
  },

  11: { // Justice
    name: 'Justice',
    upright: [
      { text: "Justice will not be served until those who are unaffected are as outraged as those who are.", source: "Benjamin Franklin" },
      { text: "The truth will set you free. But first, it will piss you off.", source: "Gloria Steinem" },
      { text: "You reap what you sow.", source: "Galatians 6:7 (Bible)" },
      { text: "The arc of the moral universe is long, but it bends toward justice.", source: "MLK Jr." },
      { text: "An eye for an eye will make the whole world blind.", source: "Gandhi" },
      { text: "Integrity is doing the right thing, even when no one is watching.", source: "C.S. Lewis" },
      { text: "Injustice anywhere is a threat to justice everywhere.", source: "MLK Jr." },
      { text: "The only thing necessary for the triumph of evil is for good men to do nothing.", source: "Edmund Burke" },
      { text: "Truth is like the sun. You can shut it out for a time, but it ain't going away.", source: "Elvis Presley" },
      { text: "Karma has no menu. You get served what you deserve.", source: "Cosmic Truth" },
      { text: "The universe doesn't owe you fairness—it owes you accuracy.", source: "Consequences Reality" },
    ],
    reversed: [
      { text: "The scales are rigged, but that doesn't mean you stop fighting.", source: "Resistance Energy" },
      { text: "Sometimes the law is wrong. Sometimes the law is the injustice.", source: "Moral Courage" },
      { text: "Being right doesn't always mean winning. And that's the hardest lesson.", source: "Justice Delayed" },
      { text: "You can't negotiate with reality. You can only accept it.", source: "Truth Surrender" },
      { text: "Unforgiveness is like drinking poison and expecting the other person to die.", source: "Buddha" },
      { text: "The truth you're avoiding is the one you most need to speak.", source: "Honesty Call" },
      { text: "You're the judge, jury, and executioner in your own story. Choose wisely.", source: "Self-Accountability" },
      { text: "Imbalance is feedback. Listen to it.", source: "Course Correction" },
      { text: "Justice delayed is justice denied.", source: "William Gladstone" },
    ]
  },

  12: { // The Hanged Man
    name: 'The Hanged Man',
    upright: [
      { text: "Turn on, tune in, drop out.", source: "Timothy Leary" },
      { text: "Let go or be dragged.", source: "Zen Proverb" },
      { text: "Sometimes you have to lose yourself to find yourself.", source: "Spiritual Paradox" },
      { text: "The wound is the place where the Light enters you.", source: "Rumi" },
      { text: "In letting go, we receive.", source: "St. Francis of Assisi" },
      { text: "Surrender is not giving up. It's giving over.", source: "Faith Distinction" },
      { text: "What if I fall? Oh, but my darling, what if you fly?", source: "Erin Hanson" },
      { text: "Not until we are lost do we begin to understand ourselves.", source: "Thoreau" },
      { text: "The obstacle is the path.", source: "Zen Teaching" },
      { text: "Sometimes the most productive thing you can do is relax.", source: "Mark Black" },
      { text: "Moksha is release. Liberation comes through surrender, not struggle.", source: "Hindu Philosophy" },
    ],
    reversed: [
      { text: "You're not hanging—you're stuck. There's a difference.", source: "Clarity Truth" },
      { text: "Martyrdom is not a personality trait.", source: "Boundary Reality" },
      { text: "Suffering for suffering's sake is not spiritual growth.", source: "Pain Purpose" },
      { text: "There's a difference between patience and paralysis.", source: "Action vs Waiting" },
      { text: "Stop waiting for permission to save yourself.", source: "Self-Rescue" },
      { text: "The universe doesn't reward self-sacrifice—it rewards self-actualization.", source: "Empowerment Truth" },
      { text: "You've been 'letting go' for months. Maybe it's time to grab hold.", source: "Action Call" },
      { text: "Surrender the outcome, not the effort.", source: "Balance Wisdom" },
      { text: "Stagnation masquerading as contemplation is still stagnation.", source: "Movement Need" },
    ]
  },

  13: { // Death
    name: 'Death',
    upright: [
      { text: "Behold, I make all things new.", source: "Revelation 21:5 (Bible)" },
      { text: "And I looked, and behold a pale horse: and his name that sat on him was Death.", source: "Revelation 6:8 (Bible)" },
      { text: "Death witches don't fear the reaper. We work with him. Transformation requires a funeral first.", source: "Death Magic" },
      { text: "The old has passed away; behold, the new has come.", source: "2 Corinthians 5:17 (Bible)" },
      { text: "Phoenix rising energy: died once, won't do it again, but absolutely will set fire to whatever tries to bury me.", source: "Rebirth Philosophy" },
      { text: "Every new beginning comes from some other beginning's end.", source: "Seneca" },
      { text: "The snake which cannot cast its skin has to die.", source: "Friedrich Nietzsche" },
      { text: "Shed your skin. Burn the old self. The witch who emerges from the ashes is the real you.", source: "Transformation Witchcraft" },
      { text: "What the caterpillar calls the end, the rest of the world calls a butterfly.", source: "Lao Tzu" },
      { text: "You can't keep dancing with the devil and wonder why you're still in hell. At some point, you have to burn the whole ballroom down.", source: "Transformation Arc" },
      { text: "Sometimes good things fall apart so better things can fall together.", source: "Marilyn Monroe" },
      { text: "The wound is where the light enters you.", source: "Rumi" },
      { text: "You must be willing to give up the life you planned to have the life that's waiting for you.", source: "Joseph Campbell" },
      { text: "For everything there is a season: a time to be born and a time to die.", source: "Ecclesiastes 3:2 (Bible)" },
    ],
    reversed: [
      { text: "The people perish for lack of vision.", source: "Proverbs 29:18 (Bible)" },
      { text: "Clinging to the past is why you can't reach the future.", source: "Letting Go Truth" },
      { text: "You can't heal in the same environment that made you sick.", source: "Recovery Wisdom" },
      { text: "Resistance to change is insistence on pain.", source: "Growth Philosophy" },
      { text: "The definition of insanity is doing the same thing over and over and expecting different results.", source: "Einstein (attributed)" },
      { text: "You're not stuck. You're just comfortable with your suffering.", source: "Brutal Truth" },
      { text: "Grief is love with nowhere to go—but you can't make a home in it forever.", source: "Mourning Wisdom" },
      { text: "Holding on to anger is like drinking poison and expecting the other person to die.", source: "Buddha" },
      { text: "You can't start the next chapter if you keep re-reading the last one.", source: "Life Coaching" },
      { text: "Fear of change is what keeps most people stuck in mediocrity.", source: "Success Philosophy" },
      { text: "Some people die at 25 and aren't buried until 75.", source: "Benjamin Franklin" },
    ]
  },

  14: { // Temperance
    name: 'Temperance',
    upright: [
      { text: "Balance is not something you find, it's something you create.", source: "Jana Kingsford" },
      { text: "The middle way is the best way.", source: "Buddha" },
      { text: "In all things, balance.", source: "Ancient Principle" },
      { text: "Moderation in all things, including moderation.", source: "Oscar Wilde" },
      { text: "The cure for anything is salt water: sweat, tears, or the sea.", source: "Isak Dinesen" },
      { text: "Water does not resist. Water flows.", source: "Margaret Atwood" },
      { text: "Be like water making its way through cracks.", source: "Bruce Lee" },
      { text: "Patience is not passive; on the contrary, it is active; it is concentrated strength.", source: "Edward G. Bulwer-Lytton" },
      { text: "The goal is to dance lightly with life, not to be dragged through it.", source: "Flow Philosophy" },
      { text: "Integration, not perfection.", source: "Modern Wellness" },
      { text: "Alchemy is the art of transformation through balance.", source: "Hermetic Wisdom" },
    ],
    reversed: [
      { text: "You can't balance what's fundamentally broken.", source: "Truth Bomb" },
      { text: "Moderation doesn't work when one side is poison.", source: "Boundary Reality" },
      { text: "Sometimes the middle ground is just a slow death.", source: "Compromise Warning" },
      { text: "You're not being balanced—you're being indecisive.", source: "Action Call" },
      { text: "Trying to please everyone means disappointing yourself.", source: "People-Pleasing Truth" },
      { text: "The scales are broken. Stop trying to fix them with wishful thinking.", source: "Reality Check" },
      { text: "Integration requires acknowledging the extremes first.", source: "Shadow Work" },
      { text: "Balance is not neutrality. It's conscious choice.", source: "Active Wisdom" },
      { text: "You're watering yourself down to make others comfortable.", source: "Authenticity Call" },
    ]
  },

  15: { // The Devil
    name: 'The Devil',
    upright: [
      { text: "Get behind me, Satan!", source: "Matthew 16:23 (Bible)" },
      { text: "Shadow work isn't 'love and light.' It's facing the parts of yourself that scare you and saying, 'You're mine too.'", source: "Shadow Witch" },
      { text: "He was chaos and she was his only calm. And somehow, that felt like the most dangerous addiction of all.", source: "Dark Romance Aesthetic" },
      { text: "Touch her and you won't have hands. Look at her wrong and you won't have eyes.", source: "Possessive MMC Energy" },
      { text: "Dark magic isn't evil. It's just magic done in the dark. The warlock who denies his shadow is the most dangerous.", source: "Warlock Philosophy" },
      { text: "I don't want easy. I want the kind of complicated that ruins me for anyone else.", source: "BookTok Dark Romance" },
      { text: "He wasn't a knight in shining armor. He was the dragon who learned to share his hoard.", source: "Morally Grey MMC" },
      { text: "The greatest trick the devil ever pulled was convincing the world he didn't exist.", source: "The Usual Suspects" },
      { text: "We have met the enemy and he is us.", source: "Pogo" },
      { text: "Some people are worth burning the world for. I just hadn't met mine yet.", source: "Villain Arc" },
      { text: "Every witch has a devil on her shoulder. Mine just happens to give better advice than my angel.", source: "Shadow Integration" },
      { text: "Evil is a point of view. God kills indiscriminately and so shall we.", source: "Anne Rice, Interview with the Vampire" },
      { text: "The chains of habit are too weak to be felt until they are too strong to be broken.", source: "Samuel Johnson" },
      { text: "Obsession dressed up as love still feels like devotion.", source: "Dark Romance Truth" },
      { text: "Hell is empty and all the devils are here.", source: "William Shakespeare" },
    ],
    reversed: [
      { text: "Submit yourselves therefore to God. Resist the devil, and he will flee from you.", source: "James 4:7 (Bible)" },
      { text: "The price of freedom is responsibility—and you're finally ready to pay it.", source: "Liberation Philosophy" },
      { text: "You taught the devil how to dance, now it's time to sit down.", source: "Recovery Energy" },
      { text: "Breaking chains isn't dramatic—it's a thousand small choices to choose yourself.", source: "Healing Truth" },
      { text: "Addiction ends when the pain of staying the same exceeds the pain of change.", source: "Recovery Wisdom" },
      { text: "The cage is open. If it were me, I'd run.", source: "Freedom Moment" },
      { text: "You don't have to set yourself on fire to keep someone else warm.", source: "Codependency Recovery" },
      { text: "Rock bottom isn't a place—it's a decision to stop digging.", source: "Turning Point" },
      { text: "The first step is admitting you have a problem. The second is not going back to it.", source: "12-Step Wisdom" },
      { text: "Freedom is what you do with what's been done to you.", source: "Jean-Paul Sartre" },
      { text: "You've been loyal to your trauma long enough. Time to be loyal to your healing.", source: "Therapy Wisdom" },
    ]
  },

  16: { // The Tower
    name: 'The Tower',
    upright: [
      { text: "Every valley shall be exalted, and every mountain and hill shall be made low.", source: "Isaiah 40:4 (Bible)" },
      { text: "She wasn't looking for a knight. She was looking for a sword.", source: "Fantasy Heroine Energy" },
      { text: "Burn it all down. Salt the earth. Start over. Some foundations deserve to crumble.", source: "Liberation Philosophy" },
      { text: "Rock bottom became the solid foundation on which I rebuilt my life.", source: "J.K. Rowling" },
      { text: "The old version of me? Dead. This one bites back.", source: "Rebirth Arc" },
      { text: "Sometimes God allows what He hates to accomplish what He loves.", source: "Joni Eareckson Tada" },
      { text: "In the midst of chaos, there is also opportunity.", source: "Sun Tzu" },
      { text: "What doesn't kill you makes you stronger. Unless it should have killed you. Then you're just traumatized.", source: "Dark Humor Truth" },
      { text: "The tower had to fall. It was built on a lie.", source: "Truth Revelation" },
      { text: "Everything I ever let go of has claw marks on it.", source: "David Foster Wallace" },
      { text: "When one door closes, another opens. But goddamn, those hallways are a bitch.", source: "Transition Reality" },
      { text: "She was a phoenix, long before she knew what that meant.", source: "Survivor Energy" },
      { text: "I am the monster you created.", source: "Frankenstein Energy" },
    ],
    reversed: [
      { text: "For I know the plans I have for you, plans to prosper you and not to harm you.", source: "Jeremiah 29:11 (Bible)" },
      { text: "You can't rebuild on a cracked foundation and expect it to hold.", source: "Construction Wisdom" },
      { text: "Avoiding the collapse doesn't make the building safe.", source: "Denial Truth" },
      { text: "The tower is coming down whether you like it or not. Your choice is whether to get out of the way.", source: "Inevitability Wisdom" },
      { text: "Clinging to the wreckage won't save you—it'll drown you.", source: "Survival Instinct" },
      { text: "The warning bells were ringing. You just turned up the music.", source: "Ignored Red Flags" },
      { text: "Resistance to change is why you keep getting the same lessons.", source: "Growth Block" },
      { text: "You can't prevent the storm, but you can choose not to build your house on sand.", source: "Matthew 7:26 Wisdom" },
      { text: "Fear of falling kept you trapped in a burning building.", source: "Paralysis Reality" },
      { text: "The catastrophe you're avoiding is less painful than the life you're choosing.", source: "Comfort Zone Truth" },
    ]
  },

  17: { // The Star
    name: 'The Star',
    upright: [
      { text: "Hope is the thing with feathers that perches in the soul.", source: "Emily Dickinson" },
      { text: "When you wish upon a star, your dreams come true.", source: "Pinocchio" },
      { text: "We are all in the gutter, but some of us are looking at the stars.", source: "Oscar Wilde" },
      { text: "Keep your eyes on the stars and your feet on the ground.", source: "Theodore Roosevelt" },
      { text: "Shoot for the moon. Even if you miss, you'll land among the stars.", source: "Norman Vincent Peale" },
      { text: "The nitrogen in our DNA, the calcium in our teeth, the iron in our blood—we are all made of star stuff.", source: "Carl Sagan" },
      { text: "It is often in the darkest skies that we see the brightest stars.", source: "Richard Evans" },
      { text: "You are not a drop in the ocean. You are the entire ocean in a drop.", source: "Rumi" },
      { text: "The universe is not outside of you. Look inside yourself; everything that you want, you already are.", source: "Rumi" },
      { text: "Every atom in your body came from a star that exploded. You are stardust.", source: "Lawrence Krauss" },
      { text: "Moksha is remembering you were never separate from the divine. You ARE the light.", source: "Spiritual Liberation" },
    ],
    reversed: [
      { text: "You can't heal in the same environment that made you sick.", source: "Recovery Wisdom" },
      { text: "Hope without action is just wishing.", source: "Reality Check" },
      { text: "Stargazing is beautiful, but you still have to walk the earth.", source: "Grounding Truth" },
      { text: "Your faith has been tested. Don't abandon it now.", source: "Perseverance Call" },
      { text: "Disillusionment is the death of false hope and the birth of wisdom.", source: "Maturity Truth" },
      { text: "You're looking for miracles while ignoring the magic you already have.", source: "Gratitude Reminder" },
      { text: "The stars are still there. You just stopped looking up.", source: "Hope Recovery" },
      { text: "Cynicism is just disappointed idealism.", source: "Perspective Shift" },
      { text: "Don't let bitterness dim your light.", source: "Shine Anyway" },
    ]
  },

  18: { // The Moon
    name: 'The Moon',
    upright: [
      { text: "She was moon-touched and shadow-kissed, wild in ways that terrified men who needed women tame.", source: "Witchy Aesthetic" },
      { text: "We meet in the woods at midnight. We draw down the moon. We remember what they tried to make us forget.", source: "Coven Energy" },
      { text: "The moon is a loyal companion. It never leaves.", source: "Tahereh Mafi" },
      { text: "Trust your intuition. It's just your ancestors whispering warnings from beyond the veil.", source: "Mystical Wisdom" },
      { text: "Every witch has a shadow self. Mine just happens to be better at magic than I am.", source: "Shadow Witch" },
      { text: "One does not become enlightened by imagining figures of light, but by making the darkness conscious.", source: "Carl Jung" },
      { text: "Scry the water under moonlight. The answers come when you stop asking with words.", source: "Divination Practice" },
      { text: "The moon doesn't ask permission to pull the tides. Neither should you.", source: "Lunar Energy" },
      { text: "Wolves and witches understand: the moon sees everything. Hide nothing.", source: "Moon Magic" },
      { text: "The cave you fear to enter holds the treasure you seek.", source: "Joseph Campbell" },
      { text: "What we don't bring into the light, controls us from the dark.", source: "Shadow Work Truth" },
      { text: "She had that unhinged look in her eye that said 'I know things you couldn't handle knowing.'", source: "Dark Feminine Energy" },
      { text: "Witchcraft is moon phases and herb bundles, yes. But mostly it's trusting yourself when everyone says you're crazy.", source: "Modern Witch Wisdom" },
    ],
    reversed: [
      { text: "Illusions are necessary to the soul.", source: "Anne Rice Aesthetic" },
      { text: "Not everything that appears in the dark is dangerous. Sometimes it's just truth without the pretty filter.", source: "Reality Unveiling" },
      { text: "Fear and intuition are not the same thing. Learn the difference.", source: "Discernment Wisdom" },
      { text: "Your anxiety is not prophecy.", source: "Mental Health Truth" },
      { text: "The monsters under the bed are just reflections you haven't integrated yet.", source: "Shadow Integration" },
      { text: "Stop romanticizing your pain. It's keeping you stuck.", source: "Healing Call" },
      { text: "Confusion is a defense mechanism. You know the truth—you just don't like it.", source: "Clarity Truth" },
      { text: "The moon shows you what's there. Don't blame her for what you see.", source: "Messenger Truth" },
      { text: "Paranoia and intuition feel the same until you check the facts.", source: "Grounding Wisdom" },
    ]
  },

  19: { // The Sun
    name: 'The Sun',
    upright: [
      { text: "Joy is the simplest form of gratitude.", source: "Karl Barth" },
      { text: "There is a crack in everything. That's how the light gets in.", source: "Leonard Cohen" },
      { text: "Keep your face always toward the sunshine—and shadows will fall behind you.", source: "Walt Whitman" },
      { text: "The sun does not abandon the moon to darkness.", source: "Brian A. McBride" },
      { text: "What sunshine is to flowers, smiles are to humanity.", source: "Joseph Addison" },
      { text: "The sun himself is weak when he first rises, and gathers strength and courage as the day gets on.", source: "Charles Dickens" },
      { text: "Turn your face to the sun and the shadows fall behind you.", source: "Maori Proverb" },
      { text: "Some people are sunshine, some people are lightning, some people are hurricanes.", source: "Weather Energy" },
      { text: "Live in the sunshine, swim the sea, drink the wild air.", source: "Ralph Waldo Emerson" },
      { text: "After darkness comes light. After winter comes spring. This is law.", source: "Natural Cycles" },
      { text: "You are allowed to be both a masterpiece and a work in progress simultaneously.", source: "Sophia Bush" },
    ],
    reversed: [
      { text: "Too much of a good thing is still too much.", source: "Excess Warning" },
      { text: "Even the sun sets in paradise.", source: "Reality Check" },
      { text: "Toxic positivity is just emotional bypassing with better PR.", source: "Authentic Feeling" },
      { text: "You can't force joy. And pretending doesn't count.", source: "Genuine Emotion" },
      { text: "Behind every perfect Instagram post is someone struggling with something.", source: "Social Media Reality" },
      { text: "Sunshine can burn too. Moderation in all things.", source: "Balance Truth" },
      { text: "Your light doesn't need to be on all the time. Rest is productive.", source: "Permission to Dim" },
      { text: "Hiding behind optimism won't solve what needs to be addressed.", source: "Avoidance Pattern" },
      { text: "The sun will rise tomorrow whether you believe it or not.", source: "Faith vs Fear" },
    ]
  },

  20: { // Judgement
    name: 'Judgement',
    upright: [
      { text: "Rise from the ashes.", source: "Phoenix Wisdom" },
      { text: "Then I saw a new heaven and a new earth, for the first heaven and the first earth had passed away.", source: "Revelation 21:1 (Bible)" },
      { text: "The time is fulfilled, and the kingdom of God is at hand; repent and believe in the gospel.", source: "Mark 1:15 (Bible)" },
      { text: "We shall not cease from exploration, and the end of all our exploring will be to arrive where we started and know the place for the first time.", source: "T.S. Eliot" },
      { text: "Until we have seen someone's darkness, we don't really know who they are. Until we have forgiven someone's darkness, we don't really know what love is.", source: "Marianne Williamson" },
      { text: "There is no coming to consciousness without pain.", source: "Carl Jung" },
      { text: "The wound is where the light enters you.", source: "Rumi" },
      { text: "Resurrection is not just for Easter. It's for every moment you choose to rise again.", source: "Renewal Philosophy" },
      { text: "You are not defined by your past. You are prepared by it.", source: "Joel Osteen" },
      { text: "The final stage before liberation: the reckoning. Moksha demands honesty.", source: "Spiritual Accounting" },
      { text: "Become who you are.", source: "Nietzsche" },
    ],
    reversed: [
      { text: "You can't have a spiritual awakening in the same consciousness that put you to sleep.", source: "Transformation Block" },
      { text: "The call is coming. Are you going to answer or let it go to voicemail again?", source: "Avoidance Pattern" },
      { text: "Self-judgment is just ego pretending to be enlightened.", source: "Spiritual Bypass" },
      { text: "You're waiting for permission that's never coming. Give it to yourself.", source: "Self-Authorization" },
      { text: "Accountability without self-compassion is just another form of abuse.", source: "Balanced Growth" },
      { text: "The resurrection requires you to actually leave the tomb.", source: "Action Needed" },
      { text: "You can't judge yourself into wholeness. Try love instead.", source: "Compassionate Path" },
      { text: "Karmic debt doesn't mean eternal punishment. It means learn the lesson and move on.", source: "Grace Truth" },
      { text: "The past is calling. Stop answering.", source: "Forward Focus" },
    ]
  },

  21: { // The World
    name: 'The World',
    upright: [
      { text: "It is finished.", source: "John 19:30 (Bible)" },
      { text: "The world is yours.", source: "Scarface" },
      { text: "And in the end, the love you take is equal to the love you make.", source: "The Beatles" },
      { text: "Everything you've been through has led to this moment.", source: "Culmination Truth" },
      { text: "You are the universe experiencing itself.", source: "Alan Watts" },
      { text: "The end is just another beginning in disguise.", source: "Cycle Wisdom" },
      { text: "Wherever you go, there you are.", source: "Buckaroo Banzai" },
      { text: "Not all who wander are lost.", source: "J.R.R. Tolkien" },
      { text: "What you seek is seeking you.", source: "Rumi" },
      { text: "The journey of a thousand miles ends with a single step too.", source: "Completion Wisdom" },
      { text: "Moksha achieved: you are free. Now what will you do with your liberation?", source: "Freedom Question" },
      { text: "To have arrived is to have already left.", source: "Eternal Journey" },
    ],
    reversed: [
      { text: "So close, yet so far.", source: "Almost Truth" },
      { text: "The last mile is the hardest.", source: "Endurance Test" },
      { text: "You've climbed the mountain but refuse to enjoy the view.", source: "Sabotage Pattern" },
      { text: "Success without fulfillment is the ultimate failure.", source: "Tony Robbins" },
      { text: "You got everything you wanted and still feel empty. That's called needing therapy, not more achievement.", source: "Inner Work Call" },
      { text: "The goal was always about who you became in the pursuit, not the prize.", source: "Journey Lesson" },
      { text: "Finishing doesn't mean you're finished. There's always another level.", source: "Growth Mindset" },
      { text: "Sometimes you have to destroy the life you planned to live the one that's waiting for you.", source: "Path Correction" },
      { text: "You made it to the top and realized it was the wrong mountain.", source: "Reorientation Needed" },
      { text: "Integration is the final boss, and you're still avoiding it.", source: "Completion Block" },
    ]
  },
};

/**
 * Get quote for card (updated to support upright/reversed)
 */
export function getCardQuote(cardIndex, quantumSeed, reversed = false) {
  const cardQuotes = CARD_QUOTES_EXPANDED[cardIndex];

  if (!cardQuotes) return null;

  const quoteArray = reversed ? cardQuotes.reversed : cardQuotes.upright;

  if (!quoteArray || quoteArray.length === 0) return null;

  const index = Math.floor((quantumSeed * quoteArray.length)) % quoteArray.length;
  return quoteArray[index];
}
