"""Synthetic contracts for the full-engine candidate and missing-stat policy."""
import sqlite3,unittest
from cfb_full_engine_stress_v1 import build,values
class Contracts(unittest.TestCase):
 def db(self):
  c=sqlite3.connect(":memory:")
  c.execute("CREATE TABLE games(game_id TEXT,season INT,week INT,home_team TEXT,away_team TEXT,home_points INT,away_points INT,neutral_site INT)")
  c.execute("CREATE TABLE player_games(game_id TEXT,season INT,week INT,player_id TEXT,player_name TEXT,position TEXT,team TEXT,opponent TEXT,carries INT,rushing_yards INT,receptions INT,receiving_yards INT,pass_attempts INT,passing_yards INT,passing_touchdowns INT,rushing_touchdowns INT,receiving_touchdowns INT)")
  for w in range(1,6):
   c.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",(str(w),2024,w,"A","B",30,20,0))
   if w!=5:
    c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(str(w),2024,w,"p","P","RB","A","B",12,60,2,15,0,0,0,0,0))
   if w==4:
    c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(str(w),2024,w,"new","N","RB","A","B",20,200,0,0,0,0,0,0,0))
  return c
 def test_future_week_cannot_change_same_week_feature(self):
  c=self.db();a,m=build(c)
  first=a["rushing_yards"][0]
  self.assertEqual(first["week"],4)
  self.assertEqual(first["pid"],"p")
  self.assertEqual(m[(2024,"rushing_yards")],[2,1])
  c.execute("UPDATE player_games SET rushing_yards=999 WHERE week=4 AND player_id='p'")
  b,_=build(c)
  self.assertEqual(first["x"],b["rushing_yards"][0]["x"])
  self.assertNotEqual(first["y"],b["rushing_yards"][0]["y"])
 def test_zero_is_not_missing(self):
  self.assertEqual(values({"carries":0,"rushing_yards":0},"rushing_yards"),(0.,0.))
  self.assertIsNone(values({"carries":None,"rushing_yards":None},"rushing_yards"))
if __name__=="__main__":unittest.main()
