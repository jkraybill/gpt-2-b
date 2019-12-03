#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        for (raw_text in ["C: THE CABINET Q: This department contains the Federal Emergency Management Agency, the Secret Service & the Coast Guard A:...","C: JEOPARDY! KEYWORDS Q: This city was home to our favorite \"German-American physicist\" for the last 20 years of his life A:...","C: TEENS IN BOOKS Q: The abduction of teenager David Balfour is central to the plot of this classic 1886 novel A:...","C: MOVIES: BEEN THERE, DONE THAT Q: Stephen Frears in 1988 & Milos Forman in 1989 used this French novel as a basis for their films A:...","C: THAWS Q: This 13-letter word for the reestablishment of good relations is apt when France makes up, as with Rwanda in 2009 A:","","C: POETS & POETRY Q: Richard Lovelace wrote, \"Stone walls do not a prison make, nor iron bars\" this A:...","C: FUN WITH OPERA Q: In \"Susannah\", a Bible-inspired opera, the elders are scandalized when they see the nude Susannah doing this outside A:...","C: MOTHER RUSSIA Q: (Cheryl of the Clue Crew reports from \"poetic\" Russia.) I'm here in the town originally called Detskoye Selo, which in 1937 was renamed for this great Russian poet A:...","C: THROUGH THE LOOKING-GLASS Q: This wonderland tea party pair reappears under the new names Hatta & Haigha A:...","C: REINVENTING THE WHEEL Q: What some call the first car was an 1886 motorized one of these, now mainly ridden by little kids A:...","C: THE GONE WITH THE WIND MOVIE MUSEUM Q: The apartment house she lived in while writing the novel is adjacent to the museum A:...","C: \"MOTOR\" HEAD Q: Sometimes surrounding a VIP, it's a procession of people traveling in cars A:","","C: COFFEE BREAK Q: In a cantata by this composer heard here, Lieschen sings of her need for coffee: A:...","C: OLD DEUTERONOMY Q: Verses from Deuteronomy are contained in the Mezuzah, which you can usually find on this part of the house A:...","C: STUPID ANSWERS Q: Since 1970, the general synod has governed this church of England A:...","C: NAME THAT MUSICAL Q: A nice, young policeman becomes a protector of a Parisian prostitute A:...","C: THE 1700s Q: In the 1740s purist teacher Muhammad al-Wahhab allied himself with a ruler on this peninsula A:...","C: CATS & DOGS Q: The Akita, \"natural dog\" of this country, is said to have been brought to U.S. by Helen Keller A:...","C: ISLAND HOPPING Q: Of the world's 10 largest islands, 3 belong all or in part to Indonesia: New Guinea, Sumatra, & this one A:...","C: YOU HAD TO EXPECT SHAKESPEARE Q: In this comedy featuring a \"fantastical Spaniard\", 4 guys take a 3-year vow of celibacy & wackiness ensues A:...","C: CLASSICAL MUSIC GOES TO THE MOVIES Q: The piece heard here was frighteningly effective in this 1979 Robert Duvall film A:...","C: VETERINARY AFFAIRS Q: The Association of Avian Veterinarians says that to protect other birds, do this to a new bird for at least 6 weeks A:...","C: PALINDROMES Q: It's usually classified into two general types: pulse and continuous wave A:...","C: EARLY AMERICANS Q: In 1803 he invited William Clark along on an enterprise fraught with fatigues, dangers & honors A:...","C: THEY CALL THE WIND... Q: Warm, dry winds called foehns include the Santa Ana of California & this one that descends the Eastern Rockies A:...","C: LETTER SYMBOLISM Q: (Jon of the Clue Crew stands in front of a monitor.) The letter \"shin\", which begins one of the Hebrew names of God, inspired this Star Trek actor to create his famous hand gesture A:...","C: A WHITE CATEGORY Q: (Sarah of the Clue Crew reports from the Steinway & Sons factory in New York.) Steinway stopped using this material in keys decades ago, & in 1993 the company patented a piano key material made of a substitute for it A:...","C: ALIAS Q: Nguyen Tat Thanh began using this alias around 1941 when he founded the Viet Minh A:...","C: WOMEN IN POLITICS Q: In January 1964 this Maine senator announced that she would seek the Republican nomination for president A:...","C: ICH BIN EIN BERLINER Q: Once upon a time (well, in 1841) these brothers moved to Berlin to teach at the university A:...","C: IN FLORIDA Q: The name of this huge Florida lake is a Seminole Indian word meaning \"plenty big water\" A:...","C: FLAGS OF THE WORLD Q: In 1903, this country, seen here, was moved from one continent to another A:...","C: \"D\" IN SCIENCE Q: (Jimmy of the Clue Crew delivers the clue from a science lab.) The melting of ice doesn't raise the level of water because ice does this to a volume of water equivalent to its own volume when it melts A:...","C: ASTRONOMY'S GREATEST HITS Q: Around 240 B.C. Chinese astronomers observed this, later named for a British astronomer A:...","C: ADJECTIVES Q: Published by Viking, these 1-volume editions of authors are so called because they can be carried around A:...","C: FONDA THE MOVIES Q: This 1981 movie, Henry Fonda's last feature, was the only one in which he appeared with Jane A:...","C: VIDEO-POURRI Q: Here's a portion of the new U.S. version of this denomination, unveiled June 12, 1997 (picture of U.S. Grant) A:...","C: SCIENTIFIC INVENTIONS Q: In 1608 its inventor offered it exclusively to the Dutch government for military use A:...","C: \"A\" IN GEOGRAPHY Q: It's home to Hampshire College & to a campus of the University of Massachusetts A:...","C: IT BORDERS ONLY ONE OTHER COUNTRY Q: The 2 countries in the Caribbean that border only one other nation, each other A:...","C: TREES! Q: Though this \"syrupy\" tree is common in the U.S., about 2/3 of its species are native to China A:...","C: WOMEN WRITERS Q: In 1899 readers awakened to \"The Awakening\", written by this woman A:...","C: MILITARY GOVERNORS Q: Benjamin Franklin \"Beast\" Butler was dismissed in 1862 as military governor of this Louisiana city A:...","C: TEMPTATIONS Q: This potato chip brand changed its slogan from \"Betcha Can't Eat Just One\" to \"Too Good to Eat Just One\" A:...","C: IT'S NOT EASY BEING TWEEN Q: Alliterative 2-word term for the actions of a tween's social group to make him or her conform A:...","C: LOUIS, LOUIS Q: In 1916 Woodrow Wilson appointed this \"People's Attorney\" to the Supreme Court A:...","C: U.S. PRESIDENTS Q: In 1890, at age 6, he and his family moved to Independence, MO., his home for most of the rest of his life A:...","C: \"EH\"? Q: This adjective is used to describe the gripping ability of the tails of New World monkeys A:...","C: ANIMALS Q: This largest wild cat of the Americas looks similar to a leopard A:...","C: THE SHIELD Q: By law, this state's seal \"shall be a shield argent charged with a pine tree with a moose at the foot of it\" A:...","C: AFRICAN-AMERICAN AUTHORS Q: The \"Renaissance\" Countee Cullen helped lead, or the area where he married W.E.B. Du Bois' daughter in 1928 A:...","C: THE NOVEL'S FIRST DRAFT? Q: \"With the first pick in the NFL draft, the Chicago Bears select...Lennie!\" George beamed. Now they could buy 200 rabbit farms! A:...","C: ARTHUR MILLER Q: Miller told this future wife, \"You're the saddest girl I've ever met\" A:...","C: MARCO POLO Q: Back in Italy people laughed at Marco's stories, like the one about Asians using this \"black stone\" as fuel A:...","C: METER MAIDS Q: This lady from an old New England family wrote \"A Lady\", which says, \"You are beautiful and faded, like an old opera tune\" A:...","C: JAZZ Q: His \"The Girl from Ipanema\" won the Grammy for Record of the Year in 1964; the first jazz record to do so A:...","C: LET US WORSHIP Q: You don't have to be from New Orleans to know that hagiolatry is the worship of or a deep reverence for these A:...","C: HISTORIC GEOGRAPHY Q: The major settlement of New Netherland was New Amsterdam on this island A:...","C: THOMAS MORE Q: This German-born artist did the 1527 painting of More seen here A:...","C: COLLEGES & UNIVERSITIES Q: Charlotte Amalie is the site of the college of this U.S. possession A:...","C: GRAY'S ANATOMY Q: The hypoglossal nerve functions as the motor to most of the muscles of this organ A:...","C: A HOST OF TV HOSTS Q: (Hi, I'm Tom Bergeron) Bob Saget was the original host & I'm the current host of this comedy TV show that made its debut in 1989 A:...","C: WASHINGTON STATE Q: 2 restaurants atop this 605-foot Seattle landmark turn at one revolution per hour A:...","C: U.S. STATE NAMES Q: Of the 4 states that begin & end with the same vowel, the one that doesn't begin & end with the same letter as the other 3 states A:...","C: THE UNIVERSE Q: Of elliptical, spiral, or irregular, shape of the Milky Way Galaxy A:...","C: HELP! Q: A resident member of a hospital's medical staff, or an unpaid trainee A:...","C: WORLD LEADERS Q: In 1979 rumors linked this 86-year-old Yugoslav leader with a 33-year-old pop singer A:...","C: GARDENING Q: If your soil is too heavy or sandy, you can add this moss to improve its texture A:...","C: METALLICA Q: In 1984 the album \"Ride the Lightning\" by Metallica achieved this status of 500,000 copies sold A:...","C: THE OLYMPIC GAMES Q: At the 2010 Vancouver games, Kim Yu-Na of this country skated away with gold & a world record in the long program A:...","C: GIVING YOU THE BIRD Q: Aka a redbird & protected by law, this bird may have up to 4 broods from April to August each year A:...","C: HOME Q: Perhaps from the Latin for \"split\", it's a thin piece of asphalt laid in overlapping rows to cover a roof A:...","C: CAROLS Q: This last name of the model & actress seen here is also on your computer keyboard A:...","C: ENTERTAINERS OF THE PAST Q: In 1980 this 84-year-old hit the pop & country charts with \"I Wish I was Eighteen Again\" A:...","C: ODE TO ENGLAND Q: In 1811 this group smashed & mangled lots of machinery of things newfangled A:...","C: GRAY'S ANATOMY Q: These nodes 0.1 centimeters to 2.5 centimeters (1 inch) long, are especially numerous in the neck A:...","C: 19th CENTURY AMERICA Q: On February 6, 1899 the Senate ratified the Treaty of Paris, ending this conflict A:...","C: ALAN Q: His character taught Abigail Breslin that dance routine for the \"Little Miss Sunshine\" pageant A:...","C: FACTS & FIGURES Q: A survey for Hebrew National found that at 68%, this is the top hot dog topping A:...","C: OBAMAMANIA Q: For their first date, Barack took Michelle to see this director's film \"Do the Right Thing\" A:...","C: WORDS FROM THE PERSIAN Q: It's a piece of cloth worn by women & draped about the head & shoulders A:...","C: WHAT'S THE BIG IDEA? Q: The political philosophy of \"minimum government, maximum freedom\", or a party whose motto is just that A:...","C: WORLD CAPITALS Q: Laid out in 1874, the botanical gardens in this capital have an observatory with a statue of Tycho Brahe A:...","C: PARTS OF A SONG Q: Also a way to end a movie scene, it's a way to end a song with a gradual decrease of volume A:...","C: NAME THE TV SHOW Q: \"Do you poop out at parties? Are you unpopular? The answer to all your problems is in this little bottle\" A:...","C: NOT LITERALLY Q: No, if you're \"literally climbing the walls\" you'd be this boyfriend of Mary Jane Watson A:...","C: DO YOU HEAR SOMETHING? Q: An exuberant cheer lends its name to this zany alliterative noisemaker A:...","C: ART CLASS Q: You're on a roll if you know in Japan makemono are horizontal paintings done on these & kakemono are vertical ones A:...","C: SMART ANSWERS Q: This adjective that means \"cunning\" probably comes from the name of a \"tamed\" insectivore A:...","C: \"W\"ORDS Q: (Jimmy of the Clue Crew presents the clue on a monitor.) Fingerprints are impressions of the friction ridges on the skin that form an overall pattern. The three main ones are loops, arches, and these A:...","C: DOROTHYS, REBECCAS & SUMMERS Q: Usually wearing very little, this famous Dorothy appeared in all the \"Road\" movies with Bob Hope & Bing Crosby A:...","C: UNUSUAL TREES Q: White fringe tree, named for its fringes of white flowers, is also called \"old man's\" this facial feature A:...","C: POLITICS Q: U.S. ambassadors to this country have included Anne Armstrong & Joseph Kennedy A:...","C: GODFATHERS OF SOLE Q: This \"Supernatural\" Latin-rock guitarist launched a line of women's shoes A:...","C: ROMEOS Q: In a 1966 Royal Ballet version, he was Romeo to longtime partner Margot Fonteyn's Juliet A:...","C: NATIONAL PARKS OF THE WORLD Q: One of this country's major recreational areas is Vitosha National Park near Sofia A:...","C: 1981 Q: Later made into a movie, this off-Broadway play by Beth Henley won a Pulitzer Prize A:...","C: TECH TONIC Q: The 2nd version of his game system has over 5,000 game titles; no news on how many Blu-ray titles for the 3rd A:...","C: INVENTORS & INVENTIONS Q: In 1974 Art Fry of 3M added a weak adhesive to his church hymnal markers, creating these \"Notes\" A:...","C: EUROPEAN HISTORY Q: This grandfather of Charlemagne nailed the Moors at the Battle of Tours in 732, checking their advance A:..."]):
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)
