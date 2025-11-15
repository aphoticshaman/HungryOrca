/**
 * CARD QUOTE DATABASE
 * Pop culture wisdom hooks for each of the 78 tarot cards
 *
 * Sources: Public domain literature, fair use brief quotes from cultural touchstones
 * All quotes properly attributed for educational/transformative use
 *
 * Format: 3-5 quotes per card, quantum-selected for variety
 */

export const CARD_QUOTES = {
  // ═══════════════════════════════════════════════════════════
  // MAJOR ARCANA
  // ═══════════════════════════════════════════════════════════

  0: { // The Fool
    name: 'The Fool',
    quotes: [
      { text: "We're all mad here.", source: "Alice's Adventures in Wonderland (Public Domain)" },
      { text: "The only way to make sense out of change is to plunge into it, move with it, and join the dance.", source: "Alan Watts" },
      { text: "Do not be too timid and squeamish about your actions. All life is an experiment.", source: "Ralph Waldo Emerson (Public Domain)" },
      { text: "A journey of a thousand miles begins with a single step.", source: "Tao Te Ching (Public Domain)" },
    ]
  },

  1: { // The Magician
    name: 'The Magician',
    quotes: [
      { text: "We are what we pretend to be, so we must be careful about what we pretend to be.", source: "Kurt Vonnegut" },
      { text: "The universe is made of stories, not of atoms.", source: "Muriel Rukeyser" },
      { text: "What you seek is seeking you.", source: "Rumi (Public Domain)" },
      { text: "As above, so below; as within, so without.", source: "Hermetic Principle (Ancient, Public Domain)" },
    ]
  },

  2: { // The High Priestess
    name: 'The High Priestess',
    quotes: [
      { text: "She remembered things her soul knew before her body was born. That's the kind of witch you can't burn.", source: "Dark Academia Magic" },
      { text: "Your visions will become clear only when you can look into your own heart.", source: "Carl Jung" },
      { text: "She kept her magic quiet. Not because she was afraid, but because she knew power doesn't need to announce itself.", source: "Mystical Energy" },
      { text: "The cave you fear to enter holds the treasure you seek.", source: "Joseph Campbell" },
      { text: "Trust the girl with books in her bag and storms in her eyes. She knows things.", source: "Witchy Reader Aesthetic" },
    ]
  },

  3: { // The Empress
    name: 'The Empress',
    quotes: [
      { text: "To plant a garden is to believe in tomorrow.", source: "Audrey Hepburn" },
      { text: "I am large, I contain multitudes.", source: "Walt Whitman (Public Domain)" },
      { text: "Nature does not hurry, yet everything is accomplished.", source: "Lao Tzu (Public Domain)" },
      { text: "Abundance is not something we acquire. It is something we tune into.", source: "Wayne Dyer" },
    ]
  },

  4: { // The Emperor
    name: 'The Emperor',
    quotes: [
      { text: "He who controls the spice controls the universe.", source: "Frank Herbert, Dune (Brief quote, fair use)" },
      { text: "The buck stops here.", source: "Harry S. Truman (Public figure statement)" },
      { text: "With great power comes great responsibility.", source: "Spider-Man principle (cultural aphorism)" },
      { text: "Order and simplification are the first steps toward the mastery of a subject.", source: "Thomas Mann" },
    ]
  },

  5: { // The Hierophant
    name: 'The Hierophant',
    quotes: [
      { text: "When the student is ready, the teacher will appear.", source: "Buddhist Proverb (Public Domain)" },
      { text: "Tradition is not the worship of ashes, but the preservation of fire.", source: "Gustav Mahler" },
      { text: "We don't receive wisdom; we must discover it for ourselves after a journey that no one can take for us.", source: "Marcel Proust" },
      { text: "The master has failed more times than the beginner has even tried.", source: "Stephen McCranie" },
    ]
  },

  6: { // The Lovers
    name: 'The Lovers',
    quotes: [
      { text: "He fell first. He fell harder. And when she finally fell? They both burned.", source: "Enemies to Lovers Trope" },
      { text: "I wasn't looking for someone to complete me. I was looking for someone who wouldn't run when they saw how whole I already was.", source: "Modern Romance Wisdom" },
      { text: "Love is or it ain't. Thin love ain't love at all.", source: "Toni Morrison, Beloved (Brief quote, fair use)" },
      { text: "We accept the love we think we deserve.", source: "Stephen Chbosky, Perks of Being a Wallflower (Brief quote, fair use)" },
      { text: "The kind of love that ruins you for anyone else. That's the only kind worth having.", source: "Romance Reader Philosophy" },
    ]
  },

  7: { // The Chariot
    name: 'The Chariot',
    quotes: [
      { text: "I must not fear. Fear is the mind-killer.", source: "Frank Herbert, Dune (Brief quote, fair use)" },
      { text: "It is not the mountain we conquer, but ourselves.", source: "Edmund Hillary" },
      { text: "The only way out is through.", source: "Robert Frost (Public Domain)" },
      { text: "Do or do not. There is no try.", source: "Yoda principle (cultural wisdom)" },
    ]
  },

  8: { // Strength
    name: 'Strength',
    quotes: [
      { text: "She's whiskey in a teacup—looks sweet, burns going down, leaves you wanting more.", source: "Strong FMC Energy" },
      { text: "I am no bird; and no net ensnares me.", source: "Charlotte Brontë, Jane Eyre (Public Domain)" },
      { text: "Soft heart. Sharp edges. Dangerous combination.", source: "Modern Heroine Archetype" },
      { text: "The most common way people give up their power is by thinking they don't have any.", source: "Alice Walker" },
      { text: "She wore her scars like wings—proof she had survived the fall and learned to fly.", source: "Survivor Philosophy" },
    ]
  },

  9: { // The Hermit
    name: 'The Hermit',
    quotes: [
      { text: "Knowing yourself is the beginning of all wisdom.", source: "Aristotle (Ancient, Public Domain)" },
      { text: "I went to the woods because I wished to live deliberately.", source: "Henry David Thoreau, Walden (Public Domain)" },
      { text: "Silence is a source of great strength.", source: "Lao Tzu (Public Domain)" },
      { text: "Not all those who wander are lost.", source: "J.R.R. Tolkien (Brief quote, fair use)" },
    ]
  },

  10: { // Wheel of Fortune
    name: 'Wheel of Fortune',
    quotes: [
      { text: "The only constant in life is change.", source: "Heraclitus (Ancient, Public Domain)" },
      { text: "This too shall pass.", source: "Persian Proverb (Public Domain)" },
      { text: "We cannot direct the wind, but we can adjust the sails.", source: "Dolly Parton" },
      { text: "Life is what happens when you're busy making other plans.", source: "John Lennon" },
    ]
  },

  11: { // Justice
    name: 'Justice',
    quotes: [
      { text: "The arc of the moral universe is long, but it bends toward justice.", source: "Martin Luther King Jr. (Public figure statement)" },
      { text: "No one can make you feel inferior without your consent.", source: "Eleanor Roosevelt" },
      { text: "Injustice anywhere is a threat to justice everywhere.", source: "Martin Luther King Jr. (Public figure statement)" },
      { text: "What you do not wish for yourself, do not do to others.", source: "Confucius (Ancient, Public Domain)" },
    ]
  },

  12: { // The Hanged Man
    name: 'The Hanged Man',
    quotes: [
      { text: "Let go or be dragged.", source: "Zen Proverb" },
      { text: "Sometimes letting things go is an act of far greater power than defending or hanging on.", source: "Eckhart Tolle" },
      { text: "In the midst of movement and chaos, keep stillness inside of you.", source: "Deepak Chopra" },
      { text: "We must be willing to let go of the life we planned so as to have the life that is waiting for us.", source: "Joseph Campbell" },
    ]
  },

  13: { // Death
    name: 'Death',
    quotes: [
      { text: "You can't keep dancing with the devil and wonder why you're still in hell. At some point, you have to burn the whole ballroom down.", source: "Transformation Arc" },
      { text: "Every new beginning comes from some other beginning's end.", source: "Seneca (Ancient, Public Domain)" },
      { text: "The snake which cannot cast its skin has to die.", source: "Friedrich Nietzsche (Public Domain)" },
      { text: "Phoenix rising energy: died once, won't do it again, but absolutely will set fire to whatever tries to bury me.", source: "Rebirth Philosophy" },
      { text: "What the caterpillar calls the end, the rest of the world calls a butterfly.", source: "Lao Tzu (Public Domain)" },
    ]
  },

  14: { // Temperance
    name: 'Temperance',
    quotes: [
      { text: "Balance is not something you find, it's something you create.", source: "Jana Kingsford" },
      { text: "The middle way is the best way.", source: "Buddha (Ancient, Public Domain)" },
      { text: "In all things, balance.", source: "Ancient principle" },
      { text: "Moderation in all things, including moderation.", source: "Oscar Wilde (Public Domain)" },
    ]
  },

  15: { // The Devil
    name: 'The Devil',
    quotes: [
      { text: "He was chaos and she was his only calm. And somehow, that felt like the most dangerous addiction of all.", source: "Dark Romance Aesthetic" },
      { text: "I'm not the villain in your story. I'm the warning you didn't heed.", source: "Morally Grey Archetype" },
      { text: "Touch her and you won't have hands. Look at her wrong and you won't have eyes.", source: "Possessive Love Interest Energy" },
      { text: "The chains of habit are too weak to be felt until they are too strong to be broken.", source: "Samuel Johnson (Public Domain)" },
      { text: "Some people are worth burning the world for. I just hadn't met mine yet.", source: "BookTok Villain Arc" },
    ]
  },

  16: { // The Tower
    name: 'The Tower',
    quotes: [
      { text: "She wasn't looking for a knight. She was looking for a sword.", source: "Fantasy Heroine Energy" },
      { text: "Burn it all down. Salt the earth. Start over. Some foundations deserve to crumble.", source: "Liberation Philosophy" },
      { text: "Rock bottom became the solid foundation on which I rebuilt my life.", source: "J.K. Rowling" },
      { text: "The old version of me? Dead. This one bites back.", source: "Rebirth Arc" },
      { text: "In the midst of chaos, there is also opportunity.", source: "Sun Tzu (Ancient, Public Domain)" },
    ]
  },

  17: { // The Star
    name: 'The Star',
    quotes: [
      { text: "Hope is the thing with feathers that perches in the soul.", source: "Emily Dickinson (Public Domain)" },
      { text: "We are all in the gutter, but some of us are looking at the stars.", source: "Oscar Wilde (Public Domain)" },
      { text: "Shoot for the moon. Even if you miss, you'll land among the stars.", source: "Norman Vincent Peale" },
      { text: "Keep your face always toward the sunshine—and shadows will fall behind you.", source: "Walt Whitman (Public Domain)" },
    ]
  },

  18: { // The Moon
    name: 'The Moon',
    quotes: [
      { text: "She was moon-touched and shadow-kissed, wild in ways that terrified men who needed women tame.", source: "Witchy Aesthetic" },
      { text: "Trust your intuition. It's just your ancestors whispering warnings from beyond the veil.", source: "Mystical Wisdom" },
      { text: "One does not become enlightened by imagining figures of light, but by making the darkness conscious.", source: "Carl Jung" },
      { text: "In the depth of winter, I finally learned that within me there lay an invincible summer.", source: "Albert Camus" },
      { text: "The moon doesn't ask permission to pull the tides. Neither should you.", source: "Lunar Energy Philosophy" },
    ]
  },

  19: { // The Sun
    name: 'The Sun',
    quotes: [
      { text: "Joy is the simplest form of gratitude.", source: "Karl Barth" },
      { text: "There is a crack in everything. That's how the light gets in.", source: "Leonard Cohen" },
      { text: "Turn your face to the sun and the shadows fall behind you.", source: "Maori Proverb" },
      { text: "To be yourself in a world that is constantly trying to make you something else is the greatest accomplishment.", source: "Ralph Waldo Emerson (Public Domain)" },
    ]
  },

  20: { // Judgment
    name: 'Judgment',
    quotes: [
      { text: "Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.", source: "Rumi (Public Domain)" },
      { text: "We are our own dragons as well as our own heroes.", source: "Tom Robbins" },
      { text: "The time is always right to do what is right.", source: "Martin Luther King Jr." },
      { text: "It is never too late to be what you might have been.", source: "George Eliot (Public Domain)" },
    ]
  },

  21: { // The World
    name: 'The World',
    quotes: [
      { text: "All endings are also beginnings. We just don't know it at the time.", source: "Mitch Albom" },
      { text: "The wound is the place where the Light enters you.", source: "Rumi (Public Domain)" },
      { text: "We shall not cease from exploration, and the end of all our exploring will be to arrive where we started and know the place for the first time.", source: "T.S. Eliot" },
      { text: "The whole is greater than the sum of its parts.", source: "Aristotle (Ancient, Public Domain)" },
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // MINOR ARCANA - WANDS (Fire: Passion, Creativity, Action)
  // ═══════════════════════════════════════════════════════════

  22: { // Ace of Wands
    name: 'Ace of Wands',
    quotes: [
      { text: "A journey of a thousand miles begins with a single step.", source: "Lao Tzu (Public Domain)" },
      { text: "The secret of getting ahead is getting started.", source: "Mark Twain" },
      { text: "Begin anywhere.", source: "John Cage" },
      { text: "You are never too old to set another goal or to dream a new dream.", source: "C.S. Lewis" },
    ]
  },

  23: { // Two of Wands
    name: 'Two of Wands',
    quotes: [
      { text: "The future belongs to those who believe in the beauty of their dreams.", source: "Eleanor Roosevelt" },
      { text: "Vision without action is merely a dream. Action without vision just passes the time.", source: "Joel A. Barker" },
      { text: "The world is a book and those who do not travel read only one page.", source: "Augustine of Hippo (Ancient)" },
    ]
  },

  24: { // Three of Wands
    name: 'Three of Wands',
    quotes: [
      { text: "A ship in harbor is safe, but that is not what ships are built for.", source: "John A. Shedd" },
      { text: "Go confidently in the direction of your dreams. Live the life you have imagined.", source: "Henry David Thoreau (Public Domain)" },
      { text: "The only impossible journey is the one you never begin.", source: "Tony Robbins" },
    ]
  },

  25: { // Four of Wands
    name: 'Four of Wands',
    quotes: [
      { text: "Home is where one starts from.", source: "T.S. Eliot" },
      { text: "Celebrate what you want to see more of.", source: "Thomas J. Peters" },
      { text: "Joy is the simplest form of gratitude.", source: "Karl Barth" },
    ]
  },

  26: { // Five of Wands
    name: 'Five of Wands',
    quotes: [
      { text: "In the middle of difficulty lies opportunity.", source: "Albert Einstein" },
      { text: "The gem cannot be polished without friction, nor man perfected without trials.", source: "Chinese Proverb" },
      { text: "If you want to go fast, go alone. If you want to go far, go together.", source: "African Proverb" },
    ]
  },

  27: { // Six of Wands
    name: 'Six of Wands',
    quotes: [
      { text: "Success is not final, failure is not fatal: it is the courage to continue that counts.", source: "Winston Churchill" },
      { text: "The only place where success comes before work is in the dictionary.", source: "Vidal Sassoon" },
      { text: "I am the master of my fate, I am the captain of my soul.", source: "William Ernest Henley (Public Domain)" },
    ]
  },

  28: { // Seven of Wands
    name: 'Seven of Wands',
    quotes: [
      { text: "Stand for something or you will fall for anything.", source: "Malcolm X" },
      { text: "The price of greatness is responsibility.", source: "Winston Churchill" },
      { text: "Do not go where the path may lead, go instead where there is no path and leave a trail.", source: "Ralph Waldo Emerson (Public Domain)" },
    ]
  },

  29: { // Eight of Wands
    name: 'Eight of Wands',
    quotes: [
      { text: "Life is what happens when you're busy making other plans.", source: "John Lennon" },
      { text: "The world is moving so fast these days that the man who says it can't be done is generally interrupted by someone doing it.", source: "Elbert Hubbard" },
      { text: "Speed is irrelevant if you are going in the wrong direction.", source: "Mahatma Gandhi" },
    ]
  },

  30: { // Nine of Wands
    name: 'Nine of Wands',
    quotes: [
      { text: "It is not the mountain we conquer, but ourselves.", source: "Edmund Hillary" },
      { text: "Courage is not the absence of fear, but rather the assessment that something else is more important.", source: "Franklin D. Roosevelt principle" },
      { text: "Fall seven times, stand up eight.", source: "Japanese Proverb" },
    ]
  },

  31: { // Ten of Wands
    name: 'Ten of Wands',
    quotes: [
      { text: "You can't pour from an empty cup.", source: "Common wisdom" },
      { text: "The only way to do great work is to love what you do, or learn to say no to what you don't.", source: "Adaptation of Steve Jobs" },
      { text: "Burnout is what happens when you try to avoid being human for too long.", source: "Michael Gungor" },
    ]
  },

  32: { // Page of Wands
    name: 'Page of Wands',
    quotes: [
      { text: "Every child is an artist. The problem is how to remain an artist once we grow up.", source: "Pablo Picasso" },
      { text: "Enthusiasm is the electricity of life.", source: "Gordon Parks" },
      { text: "The desire to create is one of the deepest yearnings of the human soul.", source: "Dieter F. Uchtdorf" },
    ]
  },

  33: { // Knight of Wands
    name: 'Knight of Wands',
    quotes: [
      { text: "Fortune favors the bold.", source: "Virgil (Ancient, Public Domain)" },
      { text: "Life is either a daring adventure or nothing at all.", source: "Helen Keller" },
      { text: "I can't change the direction of the wind, but I can adjust my sails.", source: "Jimmy Dean" },
    ]
  },

  34: { // Queen of Wands
    name: 'Queen of Wands',
    quotes: [
      { text: "She's the kind of woman who walks into a room and men either fall to their knees or run for their lives. Sometimes both.", source: "Powerful FMC Archetype" },
      { text: "I am not afraid of storms, for I am learning how to sail my ship.", source: "Louisa May Alcott (Public Domain)" },
      { text: "The question isn't who's going to let me; it's who's going to stop me.", source: "Ayn Rand" },
      { text: "Fire in her soul. Grace in her heart. Chaos in her wake.", source: "Wild Woman Energy" },
    ]
  },

  35: { // King of Wands
    name: 'King of Wands',
    quotes: [
      { text: "Leadership is not about being in charge. It's about taking care of those in your charge.", source: "Simon Sinek" },
      { text: "The task of the leader is to get their people from where they are to where they have not been.", source: "Henry Kissinger" },
      { text: "A genuine leader is not a searcher for consensus but a molder of consensus.", source: "Martin Luther King Jr." },
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // MINOR ARCANA - CUPS (Water: Emotion, Intuition, Love)
  // ═══════════════════════════════════════════════════════════

  36: { // Ace of Cups
    name: 'Ace of Cups',
    quotes: [
      { text: "The heart has its reasons of which reason knows nothing.", source: "Blaise Pascal (Public Domain)" },
      { text: "Love is the only force capable of transforming an enemy into a friend.", source: "Martin Luther King Jr." },
      { text: "Where there is love there is life.", source: "Mahatma Gandhi" },
      { text: "The best and most beautiful things in the world cannot be seen or even touched. They must be felt with the heart.", source: "Helen Keller" },
    ]
  },

  37: { // Two of Cups
    name: 'Two of Cups',
    quotes: [
      { text: "Two souls recognizing each other across lifetimes: 'Oh. It's you. It's always been you.'", source: "Fated Mates Energy" },
      { text: "When you realize you want to spend the rest of your life with somebody, you want the rest of your life to start as soon as possible.", source: "Nora Ephron" },
      { text: "Love is friendship set on fire.", source: "Jeremy Taylor (Ancient)" },
      { text: "The bond snapped into place and suddenly breathing without them felt impossible.", source: "Fantasy Romance Trope" },
    ]
  },

  38: { // Three of Cups
    name: 'Three of Cups',
    quotes: [
      { text: "Friendship is born at that moment when one person says to another: 'What! You too? I thought I was the only one.'", source: "C.S. Lewis" },
      { text: "A friend is someone who knows all about you and still loves you.", source: "Elbert Hubbard" },
      { text: "Alone we can do so little; together we can do so much.", source: "Helen Keller" },
    ]
  },

  39: { // Four of Cups
    name: 'Four of Cups',
    quotes: [
      { text: "We can complain because rose bushes have thorns, or rejoice because thorn bushes have roses.", source: "Abraham Lincoln" },
      { text: "The grass is greener where you water it.", source: "Neil Barringham" },
      { text: "He who is not contented with what he has, would not be contented with what he would like to have.", source: "Socrates (Ancient, Public Domain)" },
    ]
  },

  40: { // Five of Cups
    name: 'Five of Cups',
    quotes: [
      { text: "You can't start the next chapter of your life if you keep re-reading the last one.", source: "Michael McMillan" },
      { text: "The wound is the place where the Light enters you.", source: "Rumi (Public Domain)" },
      { text: "Crying does not indicate that you are weak. Since birth, it has always been a sign that you are alive.", source: "Charlotte Brontë (Public Domain)" },
    ]
  },

  41: { // Six of Cups
    name: 'Six of Cups',
    quotes: [
      { text: "Memory is the diary we all carry about with us.", source: "Oscar Wilde (Public Domain)" },
      { text: "The past can hurt. But the way I see it, you can either run from it or learn from it.", source: "The Lion King principle (cultural wisdom)" },
      { text: "We do not remember days, we remember moments.", source: "Cesare Pavese" },
    ]
  },

  42: { // Seven of Cups
    name: 'Seven of Cups',
    quotes: [
      { text: "Not all those who wander are lost.", source: "J.R.R. Tolkien (Brief, fair use)" },
      { text: "The cave you fear to enter holds the treasure you seek.", source: "Joseph Campbell" },
      { text: "Dreams are necessary to life.", source: "Anaïs Nin" },
    ]
  },

  43: { // Eight of Cups
    name: 'Eight of Cups',
    quotes: [
      { text: "Sometimes letting things go is an act of far greater power than defending or hanging on.", source: "Eckhart Tolle" },
      { text: "You can't reach what's in front of you until you let go of what's behind you.", source: "Unknown wisdom" },
      { text: "The art of knowing is knowing what to ignore.", source: "Rumi (Public Domain)" },
    ]
  },

  44: { // Nine of Cups
    name: 'Nine of Cups',
    quotes: [
      { text: "Gratitude turns what we have into enough.", source: "Aesop (Ancient, Public Domain)" },
      { text: "Happiness is not something ready made. It comes from your own actions.", source: "Dalai Lama" },
      { text: "The greatest wealth is to live content with little.", source: "Plato (Ancient, Public Domain)" },
    ]
  },

  45: { // Ten of Cups
    name: 'Ten of Cups',
    quotes: [
      { text: "The best thing to hold onto in life is each other.", source: "Audrey Hepburn" },
      { text: "Happiness is only real when shared.", source: "Jon Krakauer" },
      { text: "Family is not an important thing. It's everything.", source: "Michael J. Fox" },
    ]
  },

  46: { // Page of Cups
    name: 'Page of Cups',
    quotes: [
      { text: "Logic will get you from A to B. Imagination will take you everywhere.", source: "Albert Einstein" },
      { text: "The future belongs to those who believe in the beauty of their dreams.", source: "Eleanor Roosevelt" },
      { text: "Every artist was first an amateur.", source: "Ralph Waldo Emerson (Public Domain)" },
    ]
  },

  47: { // Knight of Cups
    name: 'Knight of Cups',
    quotes: [
      { text: "Follow your heart, but take your brain with you.", source: "Alfred Adler" },
      { text: "Romance is the glamour which turns the dust of everyday life into a golden haze.", source: "Elinor Glyn" },
      { text: "Love looks not with the eyes, but with the mind.", source: "William Shakespeare (Public Domain)" },
    ]
  },

  48: { // Queen of Cups
    name: 'Queen of Cups',
    quotes: [
      { text: "Compassion is the basis of morality.", source: "Arthur Schopenhauer (Public Domain)" },
      { text: "The most important thing is to try and inspire people so that they can be great in whatever they want to do.", source: "Kobe Bryant" },
      { text: "When we give cheerfully and accept gratefully, everyone is blessed.", source: "Maya Angelou" },
    ]
  },

  49: { // King of Cups
    name: 'King of Cups',
    quotes: [
      { text: "The wise man knows himself to be a fool, but the fool thinks himself to be wise.", source: "William Shakespeare (Public Domain)" },
      { text: "Between stimulus and response there is a space. In that space is our power to choose our response.", source: "Viktor Frankl" },
      { text: "Mastering others is strength. Mastering yourself is true power.", source: "Lao Tzu (Public Domain)" },
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // MINOR ARCANA - SWORDS (Air: Thought, Communication, Conflict)
  // ═══════════════════════════════════════════════════════════

  50: { // Ace of Swords
    name: 'Ace of Swords',
    quotes: [
      { text: "The truth will set you free, but first it will piss you off.", source: "Gloria Steinem" },
      { text: "Speak your mind, even if your voice shakes.", source: "Maggie Kuhn" },
      { text: "Clarity affords focus.", source: "Thomas Leonard" },
    ]
  },

  51: { // Two of Swords
    name: 'Two of Swords',
    quotes: [
      { text: "When you come to a fork in the road, take it.", source: "Yogi Berra" },
      { text: "Not deciding is a decision.", source: "Common wisdom" },
      { text: "The hardest thing to learn in life is which bridge to cross and which to burn.", source: "David Russell" },
    ]
  },

  52: { // Three of Swords
    name: 'Three of Swords',
    quotes: [
      { text: "The heart was made to be broken.", source: "Oscar Wilde (Public Domain)" },
      { text: "What hurts you, blesses you. Darkness is your candle.", source: "Rumi (Public Domain)" },
      { text: "The worst part? Watching someone become a stranger who once knew every secret corner of your soul.", source: "Heartbreak Philosophy" },
      { text: "There is a crack in everything. That's how the light gets in.", source: "Leonard Cohen" },
    ]
  },

  53: { // Four of Swords
    name: 'Four of Swords',
    quotes: [
      { text: "Rest when you're weary. Refresh and renew yourself, your body, your mind, your spirit.", source: "Ralph Marston" },
      { text: "Almost everything will work again if you unplug it for a few minutes, including you.", source: "Anne Lamott" },
      { text: "Sometimes the most productive thing you can do is relax.", source: "Mark Black" },
    ]
  },

  54: { // Five of Swords
    name: 'Five of Swords',
    quotes: [
      { text: "He who fights with monsters should be careful lest he thereby become a monster.", source: "Friedrich Nietzsche (Public Domain)" },
      { text: "An eye for an eye will only make the whole world blind.", source: "Mahatma Gandhi" },
      { text: "Winning isn't everything, but wanting to win is.", source: "Vince Lombardi" },
    ]
  },

  55: { // Six of Swords
    name: 'Six of Swords',
    quotes: [
      { text: "You can't go back and change the beginning, but you can start where you are and change the ending.", source: "C.S. Lewis" },
      { text: "Sometimes the hardest part isn't letting go but rather learning to start over.", source: "Nicole Sobon" },
      { text: "The only way out is through.", source: "Robert Frost (Public Domain)" },
    ]
  },

  56: { // Seven of Swords
    name: 'Seven of Swords',
    quotes: [
      { text: "No legacy is so rich as honesty.", source: "William Shakespeare (Public Domain)" },
      { text: "The truth is rarely pure and never simple.", source: "Oscar Wilde (Public Domain)" },
      { text: "It is better to offer no excuse than a bad one.", source: "George Washington" },
    ]
  },

  57: { // Eight of Swords
    name: 'Eight of Swords',
    quotes: [
      { text: "The only thing we have to fear is fear itself.", source: "Franklin D. Roosevelt" },
      { text: "You are confined only by the walls you build yourself.", source: "Andrew Murphy" },
      { text: "The cage is open. If it were me, I'd run.", source: "Cultural wisdom" },
    ]
  },

  58: { // Nine of Swords
    name: 'Nine of Swords',
    quotes: [
      { text: "Worry does not empty tomorrow of its sorrow, it empties today of its strength.", source: "Corrie ten Boom" },
      { text: "You can't stop the waves, but you can learn to surf.", source: "Jon Kabat-Zinn" },
      { text: "Nothing in life is to be feared, it is only to be understood.", source: "Marie Curie" },
    ]
  },

  59: { // Ten of Swords
    name: 'Ten of Swords',
    quotes: [
      { text: "Rock bottom became the solid foundation on which I rebuilt my life.", source: "J.K. Rowling" },
      { text: "This too shall pass.", source: "Persian Proverb (Public Domain)" },
      { text: "The darkest hour is just before the dawn.", source: "Thomas Fuller (Public Domain)" },
    ]
  },

  60: { // Page of Swords
    name: 'Page of Swords',
    quotes: [
      { text: "Curiosity is the wick in the candle of learning.", source: "William Arthur Ward" },
      { text: "I have no special talents. I am only passionately curious.", source: "Albert Einstein" },
      { text: "Judge a man by his questions rather than his answers.", source: "Voltaire (Public Domain)" },
    ]
  },

  61: { // Knight of Swords
    name: 'Knight of Swords',
    quotes: [
      { text: "I must not fear. Fear is the mind-killer.", source: "Frank Herbert, Dune (Brief, fair use)" },
      { text: "Act swiftly, think carefully.", source: "Ancient wisdom" },
      { text: "Well done is better than well said.", source: "Benjamin Franklin (Public Domain)" },
    ]
  },

  62: { // Queen of Swords
    name: 'Queen of Swords',
    quotes: [
      { text: "She had that unhinged look in her eye that said 'I will absolutely destroy you and look fabulous doing it.'", source: "Unhinged FMC Energy" },
      { text: "I am no bird; and no net ensnares me.", source: "Charlotte Brontë, Jane Eyre (Public Domain)" },
      { text: "Honesty is the first chapter in the book of wisdom.", source: "Thomas Jefferson" },
      { text: "Knife to throat. Crown on head. That's the vibe.", source: "Fantasy Queen Aesthetic" },
    ]
  },

  63: { // King of Swords
    name: 'King of Swords',
    quotes: [
      { text: "Justice will not be served until those who are unaffected are as outraged as those who are.", source: "Benjamin Franklin (Public Domain)" },
      { text: "The mind is everything. What you think you become.", source: "Buddha (Ancient, Public Domain)" },
      { text: "Logic is the beginning of wisdom, not the end.", source: "Cultural wisdom" },
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // MINOR ARCANA - PENTACLES (Earth: Material, Resources, Body)
  // ═══════════════════════════════════════════════════════════

  64: { // Ace of Pentacles
    name: 'Ace of Pentacles',
    quotes: [
      { text: "Every accomplishment starts with the decision to try.", source: "John F. Kennedy" },
      { text: "Opportunity is missed by most people because it is dressed in overalls and looks like work.", source: "Thomas Edison" },
      { text: "The roots of all goodness lie in the soil of appreciation.", source: "Dalai Lama" },
    ]
  },

  65: { // Two of Pentacles
    name: 'Two of Pentacles',
    quotes: [
      { text: "Balance is not something you find, it's something you create.", source: "Jana Kingsford" },
      { text: "Life is like riding a bicycle. To keep your balance, you must keep moving.", source: "Albert Einstein" },
      { text: "You will never feel truly satisfied by work until you are satisfied by life.", source: "Heather Schuck" },
    ]
  },

  66: { // Three of Pentacles
    name: 'Three of Pentacles',
    quotes: [
      { text: "Alone we can do so little; together we can do so much.", source: "Helen Keller" },
      { text: "Coming together is a beginning, staying together is progress, and working together is success.", source: "Henry Ford" },
      { text: "Quality means doing it right when no one is looking.", source: "Henry Ford" },
    ]
  },

  67: { // Four of Pentacles
    name: 'Four of Pentacles',
    quotes: [
      { text: "The more you have, the more you are occupied. The less you have, the more free you are.", source: "Mother Teresa" },
      { text: "A man is rich in proportion to the number of things he can afford to let alone.", source: "Henry David Thoreau (Public Domain)" },
      { text: "He who is not contented with what he has, would not be contented with what he would like to have.", source: "Socrates (Ancient, Public Domain)" },
    ]
  },

  68: { // Five of Pentacles
    name: 'Five of Pentacles',
    quotes: [
      { text: "Sometimes you have to kind of die inside in order to rise from your own ashes and believe in yourself.", source: "Gerard Way" },
      { text: "We must accept finite disappointment, but never lose infinite hope.", source: "Martin Luther King Jr." },
      { text: "When you reach the end of your rope, tie a knot in it and hang on.", source: "Franklin D. Roosevelt" },
    ]
  },

  69: { // Six of Pentacles
    name: 'Six of Pentacles',
    quotes: [
      { text: "We make a living by what we get, but we make a life by what we give.", source: "Winston Churchill" },
      { text: "No one has ever become poor by giving.", source: "Anne Frank" },
      { text: "The measure of life is not its duration, but its donation.", source: "Peter Marshall" },
    ]
  },

  70: { // Seven of Pentacles
    name: 'Seven of Pentacles',
    quotes: [
      { text: "Patience is not the ability to wait, but the ability to keep a good attitude while waiting.", source: "Joyce Meyer" },
      { text: "Adopt the pace of nature: her secret is patience.", source: "Ralph Waldo Emerson (Public Domain)" },
      { text: "A garden requires patient labor and attention. Plants do not grow merely to satisfy ambitions or to fulfill good intentions.", source: "Liberty Hyde Bailey" },
    ]
  },

  71: { // Eight of Pentacles
    name: 'Eight of Pentacles',
    quotes: [
      { text: "The expert in anything was once a beginner.", source: "Helen Hayes" },
      { text: "Practice isn't the thing you do once you're good. It's the thing you do that makes you good.", source: "Malcolm Gladwell" },
      { text: "Excellence is not a destination; it is a continuous journey that never ends.", source: "Brian Tracy" },
    ]
  },

  72: { // Nine of Pentacles
    name: 'Nine of Pentacles',
    quotes: [
      { text: "I am my own muse. I am the subject I know best. The subject I want to know better.", source: "Frida Kahlo" },
      { text: "The most important relationship in your life is the relationship you have with yourself.", source: "Diane Von Furstenberg" },
      { text: "She remembered who she was and the game changed.", source: "Lalah Delia" },
    ]
  },

  73: { // Ten of Pentacles
    name: 'Ten of Pentacles',
    quotes: [
      { text: "Legacy is not leaving something for people. It's leaving something in people.", source: "Peter Strople" },
      { text: "The meaning of life is to find your gift. The purpose of life is to give it away.", source: "Pablo Picasso" },
      { text: "What we do for ourselves dies with us. What we do for others and the world remains and is immortal.", source: "Albert Pine" },
    ]
  },

  74: { // Page of Pentacles
    name: 'Page of Pentacles',
    quotes: [
      { text: "An investment in knowledge pays the best interest.", source: "Benjamin Franklin (Public Domain)" },
      { text: "Tell me and I forget. Teach me and I remember. Involve me and I learn.", source: "Benjamin Franklin (Public Domain)" },
      { text: "Learning is not attained by chance, it must be sought for with ardor and attended to with diligence.", source: "Abigail Adams" },
    ]
  },

  75: { // Knight of Pentacles
    name: 'Knight of Pentacles',
    quotes: [
      { text: "It does not matter how slowly you go as long as you do not stop.", source: "Confucius (Ancient, Public Domain)" },
      { text: "Slow and steady wins the race.", source: "Aesop (Ancient, Public Domain)" },
      { text: "The man who moves a mountain begins by carrying away small stones.", source: "Confucius (Ancient, Public Domain)" },
    ]
  },

  76: { // Queen of Pentacles
    name: 'Queen of Pentacles',
    quotes: [
      { text: "A woman is the full circle. Within her is the power to create, nurture and transform.", source: "Diane Mariechild" },
      { text: "The earth is what we all have in common.", source: "Wendell Berry" },
      { text: "Caring for myself is not self-indulgence, it is self-preservation.", source: "Audre Lorde" },
    ]
  },

  77: { // King of Pentacles
    name: 'King of Pentacles',
    quotes: [
      { text: "Wealth consists not in having great possessions, but in having few wants.", source: "Epictetus (Ancient, Public Domain)" },
      { text: "The best time to plant a tree was 20 years ago. The second best time is now.", source: "Chinese Proverb" },
      { text: "Do not save what is left after spending, but spend what is left after saving.", source: "Warren Buffett" },
    ]
  },
};

/**
 * Get quantum-selected quote for a card
 */
export function getCardQuote(cardIndex, quantumSeed) {
  const cardId = cardIndex; // Or map to string ID if needed
  const cardQuotes = CARD_QUOTES[cardId];

  if (!cardQuotes || !cardQuotes.quotes || cardQuotes.quotes.length === 0) {
    return null;
  }

  const index = Math.floor((quantumSeed * cardQuotes.quotes.length)) % cardQuotes.quotes.length;
  return cardQuotes.quotes[index];
}
