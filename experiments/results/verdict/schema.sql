CREATE TABLE tpcds1t.store_sales(ss_sold_date_sk int, ss_sold_time_sk int, ss_item_sk int, ss_customer_sk int, ss_cdemo_sk int, ss_hdemo_sk int, ss_addr_sk int, ss_store_sk int, ss_promo_sk int, ss_ticket_number int, ss_quantity int, ss_wholesale_cost double, ss_list_price double, ss_sales_price double, ss_ext_discount_amt double, ss_ext_sales_price double, ss_ext_wholesale_cost double, ss_ext_list_price double, ss_ext_tax double, ss_coupon_amt double, ss_net_paid double, ss_net_paid_inc_tax double, ss_net_profit double) ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' STORED AS TEXTFILE LOCATION 'hdfs://master:9000/data/tpcds/1T/store_sales';

CREATE TABLE tpcds1t.store_sales_10g(ss_sold_date_sk int, ss_sold_time_sk int, ss_item_sk int, ss_customer_sk int, ss_cdemo_sk int, ss_hdemo_sk int, ss_addr_sk int, ss_store_sk int, ss_promo_sk int, ss_ticket_number int, ss_quantity int, ss_wholesale_cost double, ss_list_price double, ss_sales_price double, ss_ext_discount_amt double, ss_ext_sales_price double, ss_ext_wholesale_cost double, ss_ext_list_price double, ss_ext_tax double, ss_coupon_amt double, ss_net_paid double, ss_net_paid_inc_tax double, ss_net_profit double) ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' STORED AS TEXTFILE LOCATION 'hdfs://master:9000/data/tpcds/10G/store_sales';

CREATE TABLE tpcds1t.store_sales_100g(ss_sold_date_sk int, ss_sold_time_sk int, ss_item_sk int, ss_customer_sk int, ss_cdemo_sk int, ss_hdemo_sk int, ss_addr_sk int, ss_store_sk int, ss_promo_sk int, ss_ticket_number int, ss_quantity int, ss_wholesale_cost double, ss_list_price double, ss_sales_price double, ss_ext_discount_amt double, ss_ext_sales_price double, ss_ext_wholesale_cost double, ss_ext_list_price double, ss_ext_tax double, ss_coupon_amt double, ss_net_paid double, ss_net_paid_inc_tax double, ss_net_profit double) ROW FORMAT DELIMITED FIELDS TERMINATED BY '|' STORED AS TEXTFILE LOCATION 'hdfs://master:9000/data/tpcds/100G/store_sales';

10g:28800991   520k
100g:287997024 2m
1t: 2879987999 5m 	0.001736118