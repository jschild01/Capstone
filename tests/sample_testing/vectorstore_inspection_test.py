import os
import sys
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import deeplake
from langchain.schema import Document
from langchain_community.vectorstores import DeepLake
import shutil

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.logging_config import setup_logging

# Define paths
DATA_DIR = "/home/ubuntu/Capstone/data"
LOG_DIR = os.path.join(project_root, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logging()


def get_directory_size(path: str) -> float:
    """Calculate total size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024  # Convert to MB


def log_test_results(test_name: str, results: Dict[str, Any]):
    """Log test results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"vectorstore_test_{timestamp}.json")

    log_data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "results": results
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Test results logged to: {log_file}")


def inspect_dataset(dataset_path: str) -> Dict[str, Any]:
    """Inspect the DeepLake dataset contents in detail."""
    try:
        logger.info(f"\nInspecting dataset at: {dataset_path}")
        total_size = get_directory_size(dataset_path)
        logger.info(f"Total dataset size on disk: {total_size:.2f} MB")

        ds = deeplake.load(dataset_path)

        info = {
            "tensors": {},
            "total_samples": 0,
            "storage_info": {},
            "directory_structure": {}
        }

        # Get storage details
        storage_path = os.path.join(dataset_path, 'storage')
        if os.path.exists(storage_path):
            storage_files = os.listdir(storage_path)
            logger.info("\nStorage contents:")
            for file in storage_files:
                file_path = os.path.join(storage_path, file)
                size = os.path.getsize(file_path)
                logger.info(f"- {file}: {size / 1024 / 1024:.2f} MB")
                info["storage_info"][file] = size

        # Inspect tensors
        logger.info("\nTensor Details:")
        for tensor_name in ds.tensors:
            tensor = ds.tensors[tensor_name]
            shape = tensor.shape
            dtype = tensor.dtype if hasattr(tensor, 'dtype') else 'unknown'
            storage_size = 0

            # Try to get tensor storage size
            tensor_path = os.path.join(storage_path, tensor_name) if os.path.exists(storage_path) else None
            if tensor_path and os.path.exists(tensor_path):
                storage_size = get_directory_size(tensor_path)

            info["tensors"][tensor_name] = {
                "shape": shape,
                "dtype": str(dtype),
                "storage_size_mb": storage_size,
                "samples": []
            }

            logger.info(f"\nTensor: {tensor_name}")
            logger.info(f"Shape: {shape}")
            logger.info(f"Type: {dtype}")
            logger.info(f"Storage size: {storage_size:.2f} MB")

            if info["total_samples"] == 0 and len(shape) > 0:
                info["total_samples"] = shape[0]

            # Sample first few elements if tensor is not empty
            if shape and shape[0] > 0:
                try:
                    sample_size = min(2, shape[0])
                    logger.info(f"First {sample_size} elements:")
                    for i in range(sample_size):
                        sample = tensor[i].numpy()
                        if isinstance(sample, bytes):
                            sample = sample.decode('utf-8')
                        if isinstance(sample, (list, tuple)) and len(sample) > 100:
                            sample = f"{str(sample[:100])}... (truncated)"
                        logger.info(f"Element {i}: {sample}")
                        info["tensors"][tensor_name]["samples"].append(str(sample))
                except Exception as e:
                    logger.error(f"Error sampling tensor {tensor_name}: {e}")

        # Check dataset directory structure
        logger.info("\nDataset Directory Structure:")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 4 * level
            subpath = root.replace(dataset_path, '').lstrip(os.sep)
            info["directory_structure"][subpath] = {"files": {}}

            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                size = os.path.getsize(os.path.join(root, f))
                logger.info(f"{subindent}{f}: {size / 1024 / 1024:.2f} MB")
                info["directory_structure"][subpath]["files"][f] = size

        return info

    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")
        return {"error": str(e)}


def test_sample_queries(retriever: RAGRetriever) -> List[Tuple[str, int, bool, bool]]:
    """Run a series of test queries and return results with document matching.

    Returns:
        List of tuples containing (query, num_results, content_match, doc_match)
    """
    test_queries = [
        ("What did Johnny Garrison say was the exact date he started his business?", "afs21189a_en.txt",
         "March 19, 1939"),

        ("Before owning his own store, where did Johnny Garrison say he worked first?", "afs21189a_en.txt",
         "Rogers then, now Colonial"),

        ("What was Slop Jackson's memorable greeting according to the interview?", "afs21189a_en.txt",
         "Fine, and dandy myself, and I hope you're all right"),

        ("What ingredients did the Vietnamese interviewee say go into pho?", "mb_r019_01_en.txt",
         "Noodles, they put some soup in there, some beef, some vegetables"),

        ("When did Johnny Garrison purchase his store from Pittman Grocery Company?", "afs21189a_en.txt",
         "October 20, 1945"),

        ("What was the profit-sharing arrangement when Johnny Garrison managed the store for Pittman?",
         "afs21189a_en.txt", "I got 50 cents"),

        ("Why did the Vietnamese interviewee say their parents couldn't come to America?", "mb_r019_01_en.txt",
         "English, you know, second language"),

        ("What was Brown's characteristic greeting according to the interview?", "afs21189a_en.txt",
         "Good to be here. So many done been here and gone on"),

        ("Complete this sentence: 'I managed it for Pittman Grocery Company in a most unusual and'", "afs21189a_en.txt",
         "unique way"),

        (
        "Complete this sentence: 'Under the same circumstances, when you met Brown on the street, he stopped and his first word was'",
        "afs21189a_en.txt", "Good to be here"),

        (
        "Complete this sentence: 'When I see this old mill house in the bottom of the deepest part of Crystal Lake, where there's supposed to be'",
        "afs21189a_en.txt", "no bottom to it"),

        ("Complete this sentence: 'Because when the old people live in my country, they are very'", "mb_r019_01_en.txt",
         "friendship"),

        ("Complete this sentence: 'First we eat, then we do'", "dt_r018_01_en.txt", "business"),

        ("Complete this sentence: 'They're becoming more prevalent in this area than they'", "afs21189a_en.txt",
         "used to be"),

        ("Complete this sentence: 'One morning he went down to do some corn growing, and found his mill had'",
         "afs21189a_en.txt", "sunk into Crystal Lake"),

        ("Complete this sentence: 'It's the only stream that I know of in this county that runs'", "afs21218a_en.txt",
         "north"),

        ("Complete this sentence: 'They obtained my merchandise at'", "afs21189a_en.txt", "broker's cost"),

        ("Complete this sentence: 'They just want to go to their friend. They don't need to make'", "mb_r019_01_en.txt",
         "appointment"),

        ("What specific dates were the songs recorded at Akron, Ohio?", "sr28a_en.txt",
         "Sunday, June the 27th, 1937"),

        ("How does Captain Nye describe his family size on the canal?", "sr28a_en.txt",
         "There were 18 of us, 11 boys and 7 girls. And I'm the 15th youngster, 9th boy"),

        ("What does the song say about gambling locations?", "sr21a_en.txt",
         "I've gambled down in Washington, I've gambled down in Spain, I'm going down in Georgia to gamble my last game"),

        ("In Aaron's Green Shore, how are the woman's features described?", "sr21a_en.txt",
         "Her eyes were like two sparkling diamonds, as the stars of a clear frosty night, her cheeks were two blooming roses, and her teeth all the more of the ivory so white"),

        ("What happens to Lord Lovell when he returns to town?", "sr08a_en.txt",
         "And there he heard the church bells ring, and the people a-morning round"),

        ("What do they say about Lord Lovell and Lady Nancy's deaths?", "sr08a_en.txt",
         "Lady Nancy, she died as I might today, Lord Lovell, he died tomorrow. Lady Nancy, she died out of pure, pure grief, Lord Lovell, he died for sorrow"),

        ("What grows from Lady Nancy's grave?", "sr08a_en.txt",
         "And out of her bosom there grew a rose"),

        ("How is the cook's baking described in the canal boat song?", "sr13a_en.txt",
         "She's so fond of biscuits she makes them like rocks, and woe unto you if you fall in the locks. They do for a cannon, with them we kill snakes"),

        ("What instruction is given about the mules and cook?", "sr13a_en.txt",
         "Whatever you do, be sure don't forget, tap the mule gently while the cook is on deck"),

        ("How does the song describe what the Gospel Train does?", "sr26a_en.txt",
         "She crosses every trestle, through tunnels she does roar, and over tops the mountains, she's bound for Zion's shore"),

        ("What's the warning about getting tickets for the Gospel Train?", "sr26a_en.txt",
         "Oh hurry to the station, and get your ticket there. As proof of your salvation, let nothing interfere"),

        ("In the beggar song, what does the child say about their mother?", "sr24b_en.txt",
         "Mother weeps I know she's failing and may die for want of bread"),

        ("What does the penny beggar say about their home situation?", "sr24b_en.txt",
         "Father died when I a baby, mother worked, helped us a home, but in time her strength all left her"),

        ("How does the song describe the beefsteak on the canal boat?", "sr13a_en.txt",
         "Beefsteak is tough as a fighting dog's neck, and the flies they play tag with their cook on the deck"),

        ("What happens when the superintendent visits the canal boat?", "sr13a_en.txt",
         "Her hypnotic eyes and that wonderful smile would catch the old boy and he'd eat right for miles"),

        ("What does Lord Lovell say about how long he'll be gone?", "sr08a_en.txt",
         "In a year or two or three at most, I'll return to my fair Nancy"),

        ("How does Mr. Lomax describe his first meeting with Captain Nye?", "sr28a_en.txt",
         "The kind reporter brought to my room a big, breezy, wholesome, smiling man"),

        ("What does Captain Nye say about his birth?", "sr28a_en.txt",
         "I was born on a canal boat which ran from Akron to the Ohio River"),

        ("How does Captain Nye describe where he learned songs?", "sr28a_en.txt",
         "My great-grandmothers brought many folk songs from England, and we all picked up more songs in this country wherever we traveled"),

        ("In the bird song, what is the farmer doing?", "sr10a_en.txt",
         "Sing about the farmer planting corn and beans"),

        ("What does the dying cowboy say about his family?", "sr20b_en.txt",
         "My friends and relation, I left them in Boston, my parents knew not where I had roamed"),

        ("What message does the dying cowboy want sent?", "sr20b_en.txt",
         "Please write me a letter to my gray-haired mother and break the news to my sister so dear"),

        ("How is the canal cook's personality described?", "sr13a_en.txt",
         "She sure had an answer for all that was said, and if you would cross her she'd try raise the dead"),

        ("What weapons are mentioned in Lord Vannifer's confrontation?", "sr05b_en.txt",
         "If I have two swords by my side, they cost me deep in my purse, but you shall have the very best one, and I shall have the worst"),

        ("How does Lord Vannifer's servant warn of the danger?", "sr05b_en.txt",
         "He ran till he came to the riverside, then leaped in and swam. He swam till he came to the other side, then leaped out and ran"),

        ("What does the young woman say about the gambler to her mother?", "sr21a_en.txt",
         "But the love I have for this gambling man no human tongue can tell"),

        ("How is Mary described in the canal boat wedding song?", "sr25b_en.txt",
         "Her cheeks are like the red red rose and her eyes have a lovely brown, her hair is long and beautiful"),

        ("What birds are listed in the Little Birdie song?", "sr10a_en.txt",
         "Little Snowbird, Blue Bird, Blackbird, Tant, goldfish, Tanninger, meadowlark, indigo bunting, wren, Robin Redbreast and Bob White"),

        ("What happens in the Resurrection Car verse?", "sr05a_en.txt",
         "For I'm going to a place where I'll see my Savior's face when I ride up in the resurrection car"),

        ("How does the girl first meet the gambler?", "sr21a_en.txt",
         "She took me in her parlor, she cooled me with her fan, she whispered low in a mother's ears, I love that gambling man"),

        ("In the gambling song, what does her mother ask her?", "sr21a_en.txt",
         "Oh, doctor, oh, dear doctor, why do you treat me so, to leave your dear old mother and with the gambler go"),

        ("When pumping Lake Erie, what will be on the moon?", "sr27b_en.txt",
         "And when we get done, you can tell me the song will be whiskers on the moon"),

        ("What does the woman say about marrying a rich man?", "sr23b_en.txt",
         "A rich man I will never have, he's greedy and tight always needs much serve"),

        ("What does the woman say about marrying a lawyer?", "sr23b_en.txt",
         "The lawyers always after gold, sets orphan widows out in the cold"),

        ("In Barbara Allen, where does the story begin?", "sr28a_en.txt",
         "In Scarlet Town where I was born there was a fair maid dwelling"),

        ("How does Captain Nye describe where he learned religious songs?", "sr05a_en.txt",
         "Down in the Ohio River on the Kentucky side at the colored camp meetings"),

        ("What happens when you cross the cook on the canal boat?", "sr13a_en.txt",
         "And if you would cross her she'd try raise the dead"),

        ("What is special about the Gospel Train's schedule?", "sr26a_en.txt",
         "She's always true to schedule, and runs on glory time"),

        ("How does Lord Vannifer's wife try to convince the young man?", "sr05b_en.txt",
         "What if I am Lord Vannifer's wife? The lord has gone from home, he has gone to seek for Henry, King George is on his throne"),

        ("What happens to the blood in the ship carpenter song?", "sr18a_en.txt",
         "And the blood from her body like a fountain did flow"),

        ("How does the daughter describe her love for the gambler to her mother?", "sr21a_en.txt",
         "Oh mother, oh dear mother, you know I love you well, but the love I have for this gambling man no human tongue can tell"),

        ("What does Captain Nye say about his manuscripts?", "sr28a_en.txt",
         "When I first started to write down the words of the old songs, I often found that I remembered the tune but not the words"),

        ("What does the cook do while others are sleeping?", "sr13a_en.txt",
         "We all have our trouble but this rule we keep, is move about easy, the cook is asleep"),

        ("How does Captain Nye describe the importance of these songs?", "sr28a_en.txt",
         "These songs are sacred to me, they bring back memories of the silver ribbon, the Ohio Canal"),

        ("What happens if you fall in the locks according to the canal song?", "sr13a_en.txt",
         "Woe unto you if you fall in the locks, they do for a cannon, with them we kill snakes"),

        ("How does the song describe the cook's pies?", "sr13a_en.txt",
         "Her pies are like leather, you need teeth of steel"),

        ("What warning is given to girls in the daughter's song?", "sr08a_en.txt",
         "Oh girls here take warning, behold my poor daughter, who met loved a stranger so cunning and wise, he betrayed and soon left her, as he did some others, then in her anguish she weakened and died"),

        ("How does the dying cowboy want his final moments handled?", "sr20b_en.txt",
         "Now take me to the graveyard and place this odd army and play the dead march as I they carry me on, just beat the drum or me and play the fight slowly for I'm a dead cowboy I know I've done wrong"),

        ("What does Lord Vannifer say about using the swords?", "sr05b_en.txt",
         "And you shall strike the very first blow, but strike it like a man, and I shall strike the very next blow, and I'll kill you if I can"),

        ("How does Captain Nye describe his family's canal boat life?", "sr28a_en.txt",
         "And we had one great old time, swimming, falling overboard, and as you might expect from a large family. And music and so on more or less controlled our home"),

        ("What does the song say about where he met the pretty gal?", "sr21a_en.txt",
         "I had been in Washington many more weeks than three, when I fell in love with a pretty little gal and she fell in love with me"),

        ("How does the Gospel Train song describe its schedule and reliability?", "sr26a_en.txt",
         "She's always true to schedule, and runs on glory time. All other routes are failure, so take this gospel line"),

        ("What happens to McGrew after he strikes Lord Vannifer?", "sr05b_en.txt",
         "McGrew he struck the very first blow, and wounded Lord Vannifer's sword, Lord Vannifer struck the very next blow, and McGrew could strike no more"),

        ("What does Lord Vannifer do after killing McGrew?", "sr05b_en.txt",
         "Then he took her round away so small, and kisses gave her three, and in his right held a sword, and pierced her fair body"),

        ("When Mrs. Waterhouse introduced Captain Nye, what did he say about himself?", "sr28a_en.txt",
         "I'm the last of the Ohio Canal captains"),

        ("What was unique about the manuscript scrolls Captain Nye brought?", "sr28a_en.txt",
         "Each scroll made up of 20 or 30 long, yellow sheets of paper pasted end to end together"),

        ("How does the woman by Lake Erie describe her love?", "sr27b_en.txt",
         "For I love the old towpath and all that things that float, so you cannot make a wagon of my old canal boat"),

        ("What does the beggar child say happened to their father?", "sr24b_en.txt",
         "Father died when I a baby, mother worked, helped us a home"),

        ("How is the journey of the Gospel Train described?", "sr26a_en.txt",
         "She runs through valleys, prairies, is safe at curves or fills. Your fare is paid, so catch her, for two so ever will"),

        ("What does the gambling man say about leaving?", "sr21a_en.txt",
         "If you ever see me coming back, I'll be with this gambling man"),

        ("What does the canal boat song say about the water pail?", "sr13a_en.txt",
         "The water pale often, you know, would get dry. She'd open the window, dip up a supply"),

        ("What happens after Lord Lovell and Lady Nancy die?", "sr08a_en.txt",
         "Lady Nancy was laid, laid in the church, Lord Lovell was laid set by her side"),

        ("What does the young woman in Aaron's Green Shore say about her mission?", "sr21a_en.txt",
         "I've come to awaken my brethren that slumber on Aaron's Green Shore"),

        ("How does the song describe the storm at sea?", "sr25a_en.txt",
         "To be scattered by the blast. And to hear the rallying trumpet thunder cut away the mast. We were crowded in the cabin not a soul would dare to sleep. While the hungry sea was roaring and the storm was on the deep"),

        ("What does the dying cowboy request about his funeral march?", "sr20b_en.txt",
         "Just beat the drum or me and play the fight slowly for I'm a dead cowboy I know I've done wrong"),

        ("What does Lord Vannifer say about giving McGrew a sword?", "sr05b_en.txt",
         "If I have two swords by my side, they cost me deep in my purse, but you shall have the very best one, and I shall have the worst"),

        ("How does Captain Nye describe where he was raised?", "sr28a_en.txt",
         "Yes, sir, I was born there and raised there... Chillicothe, Ohio, Upper Payne Street Bridge on a canal boat"),

        ("What does the woman who won't marry say about the grocer?", "sr23b_en.txt",
         "The grocer butcher gives short weight, farmer's milk is thirsty, takes water straight"),

        ("What do they say about the canal cook's soup?", "sr13a_en.txt",
         "Her soups are the limit, oh yes, all are chow. We put it away, but I won't tell you how"),

        ("How does the Gospel Train song describe who can board?", "sr26a_en.txt",
         "Oh, the sin-sick, lost, and dying, we always take them in, no matter who or where they are, if they call unto Him"),

        ("What do the lock tenders do when they see the cook coming?", "sr13a_en.txt",
         "The lock tenders run when they pull in the locks"),

        ("What does the song say about who's proud of 'Papa's new bride'?", "sr10a_en.txt",
         "Ten little devils came set by her side, said they were all proud of papa's new bride"),

        ("What does Lady Nancy's death make Lord Lovell do?", "sr08a_en.txt",
         "So he ordered the grave to be open wide"),

        ("According to Mr. Lomax, how long had he been collecting songs?", "sr28a_en.txt",
         "For more than 30 years, Mr. Lomax has been gathering the songs of the American people"),

        ("How does Mr. Lomax describe where he collected songs from?", "sr28a_en.txt",
         "To collect these songs, Mr. Lomax has ridden night herd with cowboys, visited ballad singers far back in the mountains, gone to dances in the country, spoken to tenant farmers in the deep south"),

        ("What does Captain Nye say about remembering song words?", "sr28a_en.txt",
         "When I first started to write down the words of the old songs, I often found that I remembered the tune but not the words. I'd keep humming the tunes till all of a sudden the old words would bob up"),

        ("What does Lord Vannifer say about killing the couple?", "sr05b_en.txt",
         "There now I kill that fair young man, that Scotland could afford, likewise the fairest lady, that ere the sun shone on"),

        ("What does the beggar child say happened to their mother?", "sr24b_en.txt",
         "In time her strength all left her, hunger often to us comes"),

        ("What happens if someone crosses the canal boat cook?", "sr13a_en.txt",
         "And if you would cross her she'd try raise the dead"),

        ("How does the song describe when Lord Lovell returned?", "sr08a_en.txt",
         "But he had not been gone for years or a day, strange countries for to see, when languishing thoughts came to his mind, Lady Nancy Bell he would see"),

        ("What does the gambling man say about his gambling locations?", "sr21a_en.txt",
         "I've gambled down in Washington, I've gambled down in Spain, I'm going down in Georgia to gamble my last game"),

        ("According to the dying cowboy, who else would weep for him?", "sr20b_en.txt",
         "But there is another as dear as my mother who'd weep if she knew I was dying out here"),

        ("What does Captain Nye say made his family special?", "sr28a_en.txt",
         "My family was a singing botch"),

        ("How does the song describe the train's reliability?", "sr26a_en.txt",
         "She's always true to schedule, and runs on glory time"),

        ("What details does the captain give about his birth?", "sr28a_en.txt",
         "I was born on a canal boat which ran from Akron to the Ohio River, down which we floated our cargo to Louisville, Kentucky"),

        ("In the Gospel Train song, what should you do at the station?", "sr26a_en.txt",
         "Oh hurry to the station, and get your ticket there. As proof of your salvation, let nothing interfere"),

        ("What does Lord Vannifer say about his final wound?", "sr05b_en.txt",
         "And soon you'll see there will be three, for fatal is my wound"),

        ("What does the cook do with the biscuits according to the song?", "sr13a_en.txt",
         "She's so fond of biscuits she makes them like rocks, and woe unto you if you fall in the locks. They do for a cannon, with them we kill snakes"),

        ("How does the girl in Aaron's Green Shore introduce herself?", "sr21a_en.txt",
         "I'm the daughter of Daniel O'Connell... from England I lately came o'er"),

        ("What does Captain Nye say about the importance of canal life?", "sr28a_en.txt",
         "That was the best life a man ever had"),

        ("What does Captain Nye say about learning songs from his mother?", "sr11a_en.txt",
         "My mother used to sing it to me, and the rest of us children when we were small would gather around her knee on the deck of a canal boat, which is our home"),

        ("When asked about his family on the canal, what details does Captain Nye provide?", "sr28a_en.txt",
         "Well, there were a large family, there were 18 of us, 11 boys and 7 girls. And I'm the 15th youngster, 9th boy. And we had one great old time, swimming, falling overboard, and as you might expect from a large family. And music and so on more or less controlled our home"),

        ("What happens when Mr. Frog meets Miss Mouse?", "sr11a_en.txt",
         "One night Miss Mouse, he heard his call, he went to see, but in did fall. He went for her with all his might, said, you must be my bride tonight"),

        ("How does Miss Mouse respond to Mr. Frog's marriage proposal?", "sr11a_en.txt",
         "She said, I came to hear you sing, but pussy at me made a spring. I lost my balance, then I fell, so this the truth to you I tell"),

        ("What does Lord Vannifer say about the fighting conditions?", "sr05b_en.txt",
         "Now rise, rise, rise, young man, said he, and do the best you can, for I would not have said in fair Scotland that I kill you a defenseless man"),

        ("What warning does Mother Nature give in the canal song?", "sr13a_en.txt",
         "The water pale often, you know, would get dry. She'd open the window, dip up a supply. In the food you could taste it the captain would rear. But one look from her and the weather was fair"),

        ("What does Captain Nye say was the purpose of recording these songs?", "sr28a_en.txt",
         "For the benefit and perpetual use of the Library of Congress in Washington"),

        ("How does Aaron's Green Shore describe the woman's clothing?", "sr21a_en.txt",
         "She resembled the goddess of freedom, and green was the mantle she wore, bound round with the shamrock and roses that grew along Aaron's Green Shore"),

        ("What request does the dying cowboy make about his funeral march?", "sr20b_en.txt",
         "Now take me to the graveyard and place this odd army and play the dead march as I the carry me on. Just beat the drum or me and play the fight slowly for I'm a dead cowboy I know I've done wrong"),

        ("What does the canal boat song say about running at night?", "sr13a_en.txt",
         "Whatever the weather, we'd run night or day")
    ]

    results = []
    for query, expected_file, expected_content in test_queries:
        try:
            logger.info("\n" + "=" * 50)
            logger.info(f"Query: {query}")
            logger.info(f"Expected file: {expected_file}")
            logger.info(f"Expected content: {expected_content}")

            try:
                embedding_vector = retriever.embeddings.embed_query(query)
                logger.info(f"Generated query embedding shape: {len(embedding_vector)}")
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                continue

            try:
                # Run search with boosting
                search_results = retriever.search_vector_store(query, top_k=5000)  # txt_boost=1.2 is default now
                logger.info(f"Search completed. Found {len(search_results)} results")

                # Check for content match
                content_match = any(expected_content.lower() in doc.page_content.lower()
                                    for doc in search_results)

                # Check for document match and track boosting effects
                doc_match = False
                correct_doc_rank = None

                logger.info("\nTop 5 Results:")
                for i, doc in enumerate(search_results[:5], 1):
                    original_filename = doc.metadata.get('original_filename', '')
                    file_type = doc.metadata.get('file_type', 'unknown')
                    base_score = doc.metadata.get('similarity_score', 0)
                    adjusted_score = doc.metadata.get('adjusted_similarity_score', 0)

                    # Track if this is the expected document
                    if original_filename == expected_file:
                        doc_match = True
                        correct_doc_rank = i

                    # Log result with boosting information
                    logger.info(f"\nResult {i}:")
                    logger.info(f"File: {original_filename}")
                    logger.info(f"Type: {file_type}")
                    logger.info(f"Base similarity: {base_score:.4f}")
                    logger.info(f"Adjusted similarity: {adjusted_score:.4f}")
                    if file_type == 'text':
                        logger.info(f"Boosted: Yes (20% boost applied)")
                    else:
                        logger.info("Boosted: No")

                    # Show if this contains the expected content
                    contains_expected = expected_content.lower() in doc.page_content.lower()
                    is_correct_doc = original_filename == expected_file
                    logger.info(f"Contains expected content: {contains_expected}")
                    logger.info(f"Is correct document: {is_correct_doc}")
                    logger.info(f"Content preview: {doc.page_content[:200]}...")

                results.append((query, len(search_results), content_match, doc_match))

                # Log detailed results
                logger.info("\nSearch Results Analysis:")
                logger.info(f"Content found: {content_match}")
                logger.info(f"Correct document found: {doc_match}")
                if correct_doc_rank:
                    logger.info(f"Correct document rank: {correct_doc_rank}")

            except Exception as e:
                logger.error(f"Error during similarity search: {e}")
                results.append((query, 0, False, False))

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            results.append((query, 0, False, False))

    # Calculate and log summary statistics
    total_queries = len(results)
    content_matches = sum(1 for _, _, content_match, _ in results if content_match)
    doc_matches = sum(1 for _, _, _, doc_match in results if doc_match)

    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"Content matches: {content_matches} ({content_matches / total_queries * 100:.1f}%)")
    logger.info(f"Document matches: {doc_matches} ({doc_matches / total_queries * 100:.1f}%)")
    logger.info(
        f"Perfect matches (both): {sum(1 for _, _, c, d in results if c and d)} ({sum(1 for _, _, c, d in results if c and d) / total_queries * 100:.1f}%)")

    return results

def verify_dataset_integrity(dataset_path: str) -> bool:
    """Verify the integrity of the dataset."""
    try:
        logger.info(f"\nVerifying dataset integrity at: {dataset_path}")

        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.error("Dataset path does not exist")
            return False

        # Load dataset
        ds = deeplake.load(dataset_path)

        # Check required tensors
        required_tensors = ['embedding', 'text', 'metadata']
        for tensor in required_tensors:
            if tensor not in ds.tensors:
                logger.error(f"Missing required tensor: {tensor}")
                return False
            if len(ds.tensors[tensor]) == 0:
                logger.error(f"Tensor {tensor} is empty")
                return False

        # Check consistency
        sizes = [len(ds.tensors[tensor]) for tensor in required_tensors]
        if len(set(sizes)) != 1:
            logger.error(f"Inconsistent tensor sizes: {dict(zip(required_tensors, sizes))}")
            return False

        # Sample check
        try:
            sample_idx = 0
            embedding = ds.embedding[sample_idx].numpy()
            text = ds.text[sample_idx].numpy()
            metadata = ds.metadata[sample_idx].numpy()

            logger.info("\nSample check results:")
            logger.info(f"Embedding shape: {embedding.shape}")
            logger.info(f"Text sample: {text[:100]}...")
            logger.info(f"Metadata sample: {metadata}")

            return True

        except Exception as e:
            logger.error(f"Error during sample check: {e}")
            return False

    except Exception as e:
        logger.error(f"Error verifying dataset: {e}")
        return False


def test_vectorstore_contents():
    """Test the vectorstore contents and functionality with enhanced result reporting."""
    logger.info("\nStarting vectorstore content test...")

    dataset_path = os.path.join(DATA_DIR, 'testing_yet_again')
    test_results = {
        "dataset_info": None,
        "query_results": [],
        "summary_metrics": {}
    }

    try:
        # Verify dataset integrity
        if not verify_dataset_integrity(dataset_path):
            error_msg = "Dataset integrity check failed"
            logger.error(error_msg)
            return False, {"error": error_msg}

        # Inspect dataset
        logger.info("Inspecting dataset...")
        dataset_info = inspect_dataset(dataset_path)
        test_results["dataset_info"] = dataset_info

        if "error" in dataset_info:
            logger.error(f"Dataset inspection error: {dataset_info['error']}")
            return False, test_results

        total_samples = dataset_info.get("total_samples", 0)
        logger.info(f"\nTotal samples in dataset: {total_samples}")

        if total_samples == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            return False, {"error": error_msg}

        # Initialize retriever
        logger.info("\nInitializing retriever...")
        retriever = RAGRetriever(
            dataset_path=dataset_path,
            model_name='instructor',
            logger=logger
        )

        # Load existing vectorstore in read-only mode
        logger.info("Loading existing vectorstore in read-only mode...")
        retriever.vectorstore = DeepLake(
            dataset_path=dataset_path,
            embedding_function=retriever.embeddings,
            read_only=True
        )

        # Run test queries
        logger.info("\nExecuting test queries...")
        query_results = test_sample_queries(retriever)

        # Process results
        test_results["query_results"] = [
            {
                "query": query,
                "num_results": num_results,
                "content_match": content_match,
                "document_match": doc_match
            }
            for query, num_results, content_match, doc_match in query_results
        ]

        # Calculate summary metrics
        total_queries = len(query_results)
        content_matches = sum(1 for _, _, content_match, _ in query_results if content_match)
        doc_matches = sum(1 for _, _, _, doc_match in query_results if doc_match)
        perfect_matches = sum(1 for _, _, c, d in query_results if c and d)

        test_results["summary_metrics"] = {
            "total_queries": total_queries,
            "content_matches": content_matches,
            "content_match_rate": (content_matches / total_queries) * 100 if total_queries > 0 else 0,
            "document_matches": doc_matches,
            "document_match_rate": (doc_matches / total_queries) * 100 if total_queries > 0 else 0,
            "perfect_matches": perfect_matches,
            "perfect_match_rate": (perfect_matches / total_queries) * 100 if total_queries > 0 else 0
        }

        # Log results
        log_test_results("vectorstore_content_test", test_results)

        logger.info("\nTest Results Summary:")
        logger.info(f"Total queries: {total_queries}")
        logger.info(
            f"Content matches: {content_matches} ({test_results['summary_metrics']['content_match_rate']:.1f}%)")
        logger.info(f"Document matches: {doc_matches} ({test_results['summary_metrics']['document_match_rate']:.1f}%)")
        logger.info(
            f"Perfect matches: {perfect_matches} ({test_results['summary_metrics']['perfect_match_rate']:.1f}%)")

        return True, test_results

    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        test_results["error"] = str(e)
        return False, test_results

    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        test_results["error"] = str(e)
        return False, test_results


if __name__ == "__main__":
    logger.info("Starting vectorstore tests...")
    logger.info("=" * 80)

    success, results = test_vectorstore_contents()

    if success:
        logger.info("\nVectorstore content test completed successfully.")

        if "dataset_info" in results:
            logger.info(f"Dataset size: {results['dataset_info'].get('total_samples', 0)} samples")

        if "query_results" in results:
            logger.info("\nDetailed Query Results:")
            for result in results["query_results"]:
                logger.info("\n" + "=" * 50)
                logger.info(f"Query: {result['query']}")
                logger.info(f"Results found: {result['num_results']}")
                logger.info(f"Content match: {'Yes' if result['content_match'] else 'No'}")
                logger.info(f"Document match: {'Yes' if result['document_match'] else 'No'}")

                # Add detailed result information if available
                if 'top_results' in result:
                    logger.info("\nTop 3 Results:")
                    for i, doc_info in enumerate(result['top_results'][:3], 1):
                        logger.info(f"\nResult {i}:")
                        logger.info(f"Filename: {doc_info.get('filename', 'Unknown')}")
                        logger.info(f"Content preview: {doc_info.get('content_preview', '')[:200]}...")
                        logger.info("Metadata:")
                        for key, value in doc_info.get('metadata', {}).items():
                            logger.info(f"  {key}: {value}")
                        logger.info(f"Contains expected content: {doc_info.get('contains_expected', False)}")
                        logger.info(f"Is correct document: {doc_info.get('is_correct_doc', False)}")

        if "summary_metrics" in results:
            logger.info("\nOverall Performance:")
            logger.info(f"Total queries: {results['summary_metrics']['total_queries']}")
            logger.info(f"Content match rate: {results['summary_metrics']['content_match_rate']:.1f}%")
            logger.info(f"Document match rate: {results['summary_metrics']['document_match_rate']:.1f}%")
            logger.info(f"Perfect match rate: {results['summary_metrics']['perfect_match_rate']:.1f}%")
    else:
        logger.error("\nVectorstore content test failed.")
        if "error" in results:
            logger.error(f"Error message: {results['error']}")

    logger.info("\nTest execution completed. Check the logs directory for detailed results.")